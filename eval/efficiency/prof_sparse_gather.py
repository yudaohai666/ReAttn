"""Is reuse_v1's sparse path latency-bound (gather breaks the load->compute pipeline)
or compute-bound (MMA-saturated)? Decisive test: three byte-identical kernels that
differ ONLY in how the selected k-block id is obtained.

  IDX_SCATTER   : kb = load(sel[i])   with scattered causal block ids   (real case)
  IDX_CONTIG    : kb = load(sel[i])   with sel[i] = i (contiguous ids)  (locality only)
  STATIC_CONTIG : kb = i              no data-dependent load            (pipelineable)

Same cnt per q-block => identical compute. Deltas isolate the cost of the
data-dependent address:
  SCATTER vs CONTIG  -> memory-locality (L2/coalescing) effect
  CONTIG  vs STATIC  -> pure address-indirection / pipeline-break effect
If STATIC is much faster than IDX_*, a prefetch rewrite of the real kernel has
bit-identical headroom. If all ~equal, the sparse path is at its compute ceiling.
"""
import os, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, triton, triton.language as tl

DEV = "cuda"


def _make_kernel(mode: str):
    USE_LOAD = mode in ("scatter", "contig")

    @triton.jit
    def _k(Q, K, V, O, K_sel, K_cnt,
           sbq, shq, ssq, sdq, sbk, shk, ssk, sdk,
           sbv, shv, ssv, sdv, sbo, sho, sso, sdo,
           sselz, sselq, ssels, scntz, scntq,
           qo_len, kv_len, softmax_scale,
           num_kv_groups: tl.constexpr, HEAD_DIM: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
           LOGICAL_BLOCK_SIZE: tl.constexpr, MAX_SEL: tl.constexpr,
           IS_CAUSAL: tl.constexpr, USE_LOAD: tl.constexpr):
        NSUB: tl.constexpr = LOGICAL_BLOCK_SIZE // BLOCK_N
        pid_seq = tl.program_id(0)
        pid_h = tl.program_id(1).to(tl.int64)
        pid_bz = tl.program_id(2).to(tl.int64)
        dtype = Q.type.element_ty
        logical_q_block = (pid_seq * BLOCK_M) // LOGICAL_BLOCK_SIZE
        offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_mask = offs_m < qo_len
        q_base = Q + pid_bz * sbq + pid_h * shq
        q = tl.load(q_base + offs_m[:, None] * ssq + offs_d[None, :] * sdq,
                    mask=q_mask[:, None], other=0.0).to(dtype)
        k_base = K + pid_bz * sbk + (pid_h // num_kv_groups) * shk
        v_base = V + pid_bz * sbv + (pid_h // num_kv_groups) * shv
        m_i = tl.zeros((BLOCK_M,), tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_M,), tl.float32) + 1.0
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        scale = softmax_scale * 1.44269504
        cnt = tl.load(K_cnt + pid_bz * scntz + logical_q_block * scntq)
        sel_base = K_sel + pid_bz * sselz + logical_q_block * sselq
        for i in range(0, MAX_SEL):
            if i < cnt:
                if USE_LOAD:
                    kb = tl.load(sel_base + i * ssels)
                else:
                    kb = i
                for sub in tl.static_range(NSUB):
                    kv_seq_start = kb * LOGICAL_BLOCK_SIZE + sub * BLOCK_N
                    offs_n = kv_seq_start + tl.arange(0, BLOCK_N)
                    n_mask = offs_n < kv_len
                    k = tl.load(k_base + offs_n[:, None] * ssk + offs_d[None, :] * sdk,
                                mask=n_mask[:, None], other=0.0).to(dtype)
                    qk = tl.dot(q, tl.trans(k)) * scale
                    bad = offs_n[None, :] >= kv_len
                    if IS_CAUSAL:
                        bad |= offs_m[:, None] < offs_n[None, :]
                    qk = qk + tl.where(bad, -1e6, 0.0)
                    local_m = tl.max(qk, 1)
                    m_ij = tl.maximum(m_i, local_m)
                    qk -= m_ij[:, None]
                    p = tl.math.exp2(qk)
                    l_ij = tl.sum(p, 1)
                    alpha = tl.math.exp2(m_i - m_ij)
                    acc = acc * alpha[:, None]
                    v = tl.load(v_base + offs_n[:, None] * ssv + offs_d[None, :] * sdv,
                                mask=n_mask[:, None], other=0.0).to(dtype)
                    acc += tl.dot(p.to(dtype), v)
                    l_i = l_i * alpha + l_ij
                    m_i = m_ij
        acc = acc / l_i[:, None]
        o_base = O + pid_bz * sbo + pid_h * sho
        tl.store(o_base + offs_m[:, None] * sso + offs_d[None, :] * sdo,
                 acc.to(dtype), mask=q_mask[:, None])
    return _k


def build_sel(nqb, cap, max_sel, device, scatter: bool):
    k_cnt = torch.empty((1, nqb), dtype=torch.int32, device=device)
    k_sel = torch.zeros((1, nqb, max_sel), dtype=torch.int32, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    for qb in range(nqb):
        cnt = min(qb + 1, cap)
        k_cnt[0, qb] = cnt
        if scatter and qb + 1 > cnt:
            # pick `cnt` distinct causal-valid blocks scattered across 0..qb
            perm = torch.randperm(qb + 1, generator=g, device=device)[:cnt]
            perm, _ = torch.sort(perm)
            k_sel[0, qb, :cnt] = perm.to(torch.int32)
        else:
            k_sel[0, qb, :cnt] = torch.arange(cnt, dtype=torch.int32, device=device)
    return k_sel, k_cnt


def timed(fn, n=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(n):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n


def main(s=128 * 1024, G=4, cap=64, block_size=128, BM=128, BN=128):
    Hkv, d = 1, 128
    H = Hkv * G
    nqb = s // block_size
    max_sel = cap + 2
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(1, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    grid = (triton.cdiv(s, BM), H, 1)

    kern = _make_kernel("x")

    def run(sel, cnt, use_load):
        kern[grid](
            q, k, v, out, sel, cnt,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            sel.stride(0), sel.stride(1), sel.stride(2),
            cnt.stride(0), cnt.stride(1),
            s, s, scale,
            num_kv_groups=G, HEAD_DIM=d, BLOCK_M=BM, BLOCK_N=BN,
            LOGICAL_BLOCK_SIZE=block_size, MAX_SEL=max_sel, IS_CAUSAL=True,
            USE_LOAD=use_load, num_warps=8, num_stages=2,
        )

    sel_sc, cnt_sc = build_sel(nqb, cap, max_sel, DEV, scatter=True)
    sel_co, cnt_co = build_sel(nqb, cap, max_sel, DEV, scatter=False)

    t_scatter = timed(lambda: run(sel_sc, cnt_sc, True))
    t_contig = timed(lambda: run(sel_co, cnt_co, True))
    t_static = timed(lambda: run(sel_co, cnt_co, False))

    print(f"[s={s//1024}K G={G} cap={cap} BM={BM} BN={BN}]")
    print(f"  IDX_SCATTER   (real gather, scattered) : {t_scatter:.3f} ms")
    print(f"  IDX_CONTIG    (gather, contiguous ids) : {t_contig:.3f} ms")
    print(f"  STATIC_CONTIG (no addr load, kb=i)     : {t_static:.3f} ms")
    print(f"  locality effect  SCATTER/CONTIG = {t_scatter/t_contig:.2f}x")
    print(f"  indirection cost CONTIG/STATIC  = {t_contig/t_static:.2f}x")
    print(f"  total headroom   SCATTER/STATIC = {t_scatter/t_static:.2f}x")


if __name__ == "__main__":
    for cap in (32, 64):
        main(cap=cap)
        print()
