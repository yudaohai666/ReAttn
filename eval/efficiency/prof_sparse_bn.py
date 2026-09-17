"""Does a WIDER softmax tile amortize the online-softmax serialization?

The production sparse kernel processes one 128-key selected block per online-softmax
round (BLOCK_N=128, NSUB=1). The 37% "softmax+serialization" overhead (measured in
prof_sparse_compute.py as FULL/MMA=1.58x) is the dot1->max/exp/sum/rescale->dot2
dependency chain, run once per round. Hypothesis: a WIDER tile (process W keys per
round) does the SAME total MMA but FEWER softmax rounds -> fewer dependency stalls ->
higher tensor-core utilization. Since we already proved the gather is free
(prof_sparse_gather.py), we can test the pure tile-width effect with a CONTIGUOUS
causal key range (no scatter) -- the timing question is identical.

This is a LEGAL lever for the OUTPUT-ONLY sparse path: wider BLOCK_N changes the
online-softmax tiling -> last-bit fp noise, but the sparse path emits no score and
does not drive block selection, so fp noise is allowed. (It would NOT be legal for
the anchor path, which must keep scores bit-identical.)

Sweeps BLOCK_N in {64,128,256,512}. Same per-q-block key count (cnt*128) so MMA is
matched. If wider N is faster -> a cross-block-fused rewrite of the sparse kernel is
worth it. If flat/worse -> BN=128 is already optimal, dead end.
"""
import os, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, triton, triton.language as tl

DEV = "cuda"
H800_BF16_TFLOPS = 989.0


@triton.jit
def _k(Q, K, V, O, K_cnt,
       sbq, shq, ssq, sdq, sbk, shk, ssk, sdk,
       sbv, shv, ssv, sdv, sbo, sho, sso, sdo,
       scntz, scntq,
       qo_len, kv_len, softmax_scale,
       num_kv_groups: tl.constexpr, HEAD_DIM: tl.constexpr,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
       LOGICAL_BLOCK_SIZE: tl.constexpr, IS_CAUSAL: tl.constexpr):
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
    # cnt = number of 128-key logical blocks this q-block attends (contiguous, causal).
    cnt = tl.load(K_cnt + pid_bz * scntz + logical_q_block * scntq)
    nkeys = cnt * LOGICAL_BLOCK_SIZE
    # sweep the softmax-tile width BLOCK_N over the contiguous [0, nkeys) key range.
    n_tiles = (nkeys + BLOCK_N - 1) // BLOCK_N
    for j in range(0, n_tiles):
        kv_seq_start = j * BLOCK_N
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


def timed(fn, n=15, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(n):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n


def main(s=128 * 1024, G=4, cap=64, block_size=128, BM=128):
    BM = int(os.environ.get("PROF_BM", str(BM)))
    Hkv, d = 1, 128
    H = Hkv * G
    nqb = s // block_size
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(1, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    grid = (triton.cdiv(s, BM), H, 1)
    # cnt per q-block = min(qb+1, cap) -> matches the real budget-capped selection count.
    k_cnt = torch.empty((1, nqb), dtype=torch.int32, device=DEV)
    for qb in range(nqb):
        k_cnt[0, qb] = min(qb + 1, cap)
    total_blocks = int(k_cnt.sum().item())

    def run(BN):
        _k[grid](
            q, k, v, out, k_cnt,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            k_cnt.stride(0), k_cnt.stride(1),
            s, s, scale,
            num_kv_groups=G, HEAD_DIM=d, BLOCK_M=BM, BLOCK_N=BN,
            LOGICAL_BLOCK_SIZE=block_size, IS_CAUSAL=True,
            num_warps=8, num_stages=int(os.environ.get("PROF_STAGES", "2")),
        )

    macs = H * total_blocks * 2 * (block_size * block_size * d)
    flops = 2 * macs
    print(f"[s={s//1024}K G={G} cap={cap} BM={BM}]  total_sel_blocks/head={total_blocks}")
    base = None
    for BN in (64, 128, 256, 512):
        t = timed(lambda: run(BN))
        tfl = flops / (t / 1000) / 1e12
        if BN == 128:
            base = t
        rel = f"  ({base/t:.2f}x vs BN=128)" if base else ""
        print(f"  BLOCK_N={BN:4d} : {t:.3f} ms  -> {tfl:.0f} TFLOP/s "
              f"({100*tfl/H800_BF16_TFLOPS:.0f}% peak){rel}")


if __name__ == "__main__":
    import sys
    caps = [int(x) for x in sys.argv[1:]] or [32, 64]
    for cap in caps:
        main(cap=cap)
        print(flush=True)
