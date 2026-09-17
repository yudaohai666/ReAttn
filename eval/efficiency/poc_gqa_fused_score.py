"""PoC #2: GQA-fused anchor kernel WITH per-q-head score emission.

Extends poc_gqa_fused.py: one program handles BLOCK_M query rows for ALL G heads of a
kv-group (rows stacked in the row dim, K/V loaded ONCE per group). Now it also emits
the per-logical-block lse_mass + row lse2 scratch, exactly mirroring the production
anchor inner loop (Reuse_v1._block_sparse_score_fwd_inner lines 166-184). Because rows
are independent, each row carries its own head offset (g_of) into the [b,H,row,kblk]
Mass buffer, so the UNMODIFIED production _block_score_reduce_kernel turns it into the
same block_score, and the host-side amax-over-G selection is untouched.

Validation: under BLOCK_SPARSE_NO_AUTOTUNE=1, fused (BLOCK_N=128) vs production
block_sparse_attn_with_score must be bit-identical on both `out` AND `block_score`
(proves the fusion transform changes no method logic). Then time the full
fused+reduce path at 128K vs production with_score.
"""
import os
import math
import torch
import triton
import triton.language as tl

DEV = 'cuda'


@triton.jit
def _gqa_fused_score_fwd(
    Q, K, V, O, Mass, LSE2,
    sb_q, sh_q, ss_q, sd_q,
    sb_k, sh_k, ss_k, sd_k,
    sb_v, sh_v, ss_v, sd_v,
    sb_o, sh_o, ss_o, sd_o,
    sm_z, sm_h, sm_row, sm_blk,
    sl_z, sl_h, sl_row,
    qo_len, kv_len, softmax_scale, q_row_offset,
    G: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LOGICAL_BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hkv = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    ROWS: tl.constexpr = G * BLOCK_M
    row = tl.arange(0, ROWS)
    g_of = (row // BLOCK_M).to(tl.int64)          # head within the kv-group
    m_of = row % BLOCK_M                          # local query index in the block
    local_row = pid_m * BLOCK_M + m_of            # chunk-local query row
    pos = q_row_offset + local_row                # absolute query position
    offs_d = tl.arange(0, HEAD_DIM)
    scale = softmax_scale * 1.44269504

    qh = (pid_hkv * G + g_of)                     # absolute q-head per row
    q_row = pid_b * sb_q + qh * sh_q + pos * ss_q
    q = tl.load(Q + q_row[:, None] + offs_d[None, :] * sd_q,
                mask=pos[:, None] < qo_len, other=0.0)

    k_base = K + pid_b * sb_k + pid_hkv * sh_k
    v_base = V + pid_b * sb_v + pid_hkv * sh_v

    m_i = tl.zeros((ROWS,), tl.float32) - float("inf")
    l_i = tl.zeros((ROWS,), tl.float32) + 1.0
    acc = tl.zeros((ROWS, HEAD_DIM), tl.float32)
    blk_m = tl.zeros((ROWS,), tl.float32) - float("inf")
    blk_l = tl.zeros((ROWS,), tl.float32)

    mass_base = Mass + pid_b * sm_z.to(tl.int64) + qh * sm_h.to(tl.int64)
    mass_row = local_row * sm_row
    row_valid = pos < qo_len

    hi = tl.minimum(q_row_offset + (pid_m + 1) * BLOCK_M, kv_len)
    for kj in range(0, hi, BLOCK_N):
        offs_n = kj + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * ss_k + offs_d[None, :] * sd_k,
                    mask=offs_n[:, None] < kv_len, other=0.0)      # loaded ONCE per group
        v = tl.load(v_base + offs_n[:, None] * ss_v + offs_d[None, :] * sd_v,
                    mask=offs_n[:, None] < kv_len, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        kv_mask = (offs_n[None, :] >= kv_len) | (pos[:, None] < offs_n[None, :])
        qk = qk + tl.where(kv_mask, -1e6, 0.0)
        local_m = tl.max(qk, 1)
        m_ij = tl.maximum(m_i, local_m)
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        new_blk_m = tl.maximum(blk_m, m_ij)
        blk_l = (blk_l * tl.math.exp2(blk_m - new_blk_m)
                 + l_ij * tl.math.exp2(m_ij - new_blk_m))
        blk_m = new_blk_m
        if (kj % LOGICAL_BLOCK_SIZE) == (LOGICAL_BLOCK_SIZE - BLOCK_N):
            lse_mass = tl.where(blk_l > 0.0, blk_m + tl.math.log2(blk_l), -float("inf"))
            logical_k_block = kj // LOGICAL_BLOCK_SIZE
            tl.store(mass_base + logical_k_block * sm_blk + mass_row,
                     lse_mass, mask=row_valid)
            blk_m = tl.zeros((ROWS,), tl.float32) - float("inf")
            blk_l = tl.zeros((ROWS,), tl.float32)

    out = acc / l_i[:, None]
    o_row = pid_b * sb_o + qh * sh_o + pos * ss_o
    tl.store(O + o_row[:, None] + offs_d[None, :] * sd_o,
             out.to(O.type.element_ty), mask=pos[:, None] < qo_len)

    lse2 = m_i + tl.math.log2(l_i)
    tl.store(LSE2 + pid_b * sl_z + qh * sl_h + local_row * sl_row, lse2, mask=row_valid)


def gqa_fused_with_score(q, k, v, block_size=128, softmax_scale=None,
                         BLOCK_M=64, BLOCK_N=64, num_warps=8, num_stages=2,
                         q_chunk_blocks=None):
    from pbs_attn.baselines.Reuse_v1 import _block_score_reduce_kernel
    b, H, s, d = q.shape
    Hkv = k.shape[1]
    G = H // Hkv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    nqb = (s + block_size - 1) // block_size
    nkb = nqb

    out = torch.empty_like(q)
    block_score = torch.empty((b, H, nqb, nkb), dtype=torch.float32, device=q.device)

    if q_chunk_blocks is None or q_chunk_blocks >= nqb:
        chunk_qb = nqb
    else:
        chunk_qb = q_chunk_blocks
    chunk_rows = chunk_qb * block_size
    mass = torch.empty((b, H, chunk_rows, nkb), dtype=torch.float32, device=q.device)
    lse2 = torch.empty((b, H, chunk_rows), dtype=torch.float32, device=q.device)

    for qb0 in range(0, nqb, chunk_qb):
        qb1 = min(qb0 + chunk_qb, nqb)
        nqb_c = qb1 - qb0
        q_row_offset = qb0 * block_size
        grid = (triton.cdiv(nqb_c * block_size, BLOCK_M), Hkv, b)
        _gqa_fused_score_fwd[grid](
            q, k, v, out, mass, lse2,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            mass.stride(0), mass.stride(1), mass.stride(2), mass.stride(3),
            lse2.stride(0), lse2.stride(1), lse2.stride(2),
            s, s, softmax_scale, q_row_offset,
            G=G, HEAD_DIM=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            LOGICAL_BLOCK_SIZE=block_size, num_warps=num_warps, num_stages=num_stages,
        )
        bs_view = block_score[:, :, qb0:qb1, :]
        grid2 = (nqb_c, H, b)
        _block_score_reduce_kernel[grid2](
            mass, lse2, bs_view,
            mass.stride(0), mass.stride(1), mass.stride(2), mass.stride(3),
            lse2.stride(0), lse2.stride(1), lse2.stride(2),
            bs_view.stride(0), bs_view.stride(1), bs_view.stride(2), bs_view.stride(3),
            s, nkb, q_row_offset,
            BLOCK_M=block_size,
            BLOCK_KB=min(128, triton.next_power_of_2(nkb)),
            IS_CAUSAL=True, BLOCK_N=block_size,
        )
    return out, block_score


def timed(fn, n=20, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(n):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n


def correctness():
    from pbs_attn.baselines.Reuse_v1 import block_sparse_attn_with_score
    torch.manual_seed(0)
    b, Hkv, G, s, d = 1, 2, 4, 2048, 128
    H = Hkv * G
    bs = 128
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    nqb = s // bs
    full_bm = torch.ones((b, H, nqb, nqb), dtype=torch.bool, device=DEV)

    ref_out, ref_score = block_sparse_attn_with_score(
        q, k, v, full_bm, block_size=bs, segment_size=2048, causal=True,
        softmax_scale=scale, q_chunk_blocks=None)

    # BLOCK_N must match production's fixed config (128) for a bit-identical check.
    got_out, got_score = gqa_fused_with_score(q, k, v, block_size=bs,
                                              softmax_scale=scale, BLOCK_M=64, BLOCK_N=128)
    # The upper triangle (k-block > q-block) is never written by either path
    # (causal) — it is uninitialized torch.empty garbage. Only the causal
    # lower triangle feeds block selection, so compare that region.
    tri = torch.tril(torch.ones(nqb, nqb, dtype=torch.bool, device=DEV))[None, None]
    out_diff = (got_out.float() - ref_out.float()).abs().max().item()
    sc_diff = (got_score - ref_score).masked_fill(~tri, 0).abs().max().item()
    out_id = torch.equal(got_out, ref_out)
    sc_id = torch.equal(got_score.masked_fill(~tri, 0), ref_score.masked_fill(~tri, 0))
    print(f"[correctness s={s} H={H} Hkv={Hkv} BLOCK_N=128]")
    print(f"  out         max_abs_diff = {out_diff:.3e}   bit-identical = {out_id}")
    print(f"  score(causal) max_abs_diff = {sc_diff:.3e}   bit-identical = {sc_id}")
    return out_id and sc_id


def bench(s=128 * 1024, Hkv=1, G=4):
    from pbs_attn.baselines.Reuse_v1 import block_sparse_attn_with_score
    torch.manual_seed(0)
    b, d = 1, 128
    H = Hkv * G
    bs, seg = 128, 2048
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    nqb = s // bs
    full_bm = torch.ones((b, H, nqb, nqb), dtype=torch.bool, device=DEV)

    tP = timed(lambda: block_sparse_attn_with_score(
        q, k, v, full_bm, block_size=bs, segment_size=seg, causal=True,
        softmax_scale=scale, q_chunk_blocks=None))
    print(f"\n[bench s={s//1024}K  Hkv={Hkv} G={G} H={H}]")
    print(f"  production with_score (per-head anchor path): {tP:.3f} ms")
    best = None
    for bm, bn in [(64, 64), (64, 128), (32, 64), (32, 128), (16, 128)]:
        try:
            t = timed(lambda: gqa_fused_with_score(q, k, v, block_size=bs,
                      softmax_scale=scale, BLOCK_M=bm, BLOCK_N=bn))
            tag = "  <- bit-identical config" if bn == 128 else ""
            print(f"  gqa_fused+score BLOCK_M={bm} BLOCK_N={bn}: {t:.3f} ms  ({tP/t:.2f}x){tag}")
            if best is None or t < best[0]:
                best = (t, bm, bn)
        except Exception as e:
            print(f"  gqa_fused+score BLOCK_M={bm} BLOCK_N={bn}: FAILED {type(e).__name__}: {str(e)[:70]}")
    if best:
        print(f"  => best fused = {best[0]:.3f} ms (BM={best[1]} BN={best[2]}), speedup {tP/best[0]:.2f}x")


def bench_real(label_path, s, G=4, block_size=128, segment_size=2048, chunk_ctas=2048,
               BLOCK_M=32, BLOCK_N=128):
    """Real anchor-path comparison: replicate production's per-layer anchor launch
    (all nf_kv anchor kv-heads of a layer batched, q-chunked exactly like
    reuse_v1_layer_per_hkv) using the trained label.pt, at sequence length s.
    Time depends only on (nf_kv, s), so time one representative per distinct nf_kv
    and weight by how many layers have it."""
    import math as _m
    from collections import Counter
    from pbs_attn.baselines.Reuse_v1 import block_sparse_attn_with_score
    if os.environ.get("BLOCK_SPARSE_NO_AUTOTUNE", "false").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "bench_real must run with autotune ON. BLOCK_SPARSE_NO_AUTOTUNE=1 forces "
            "production into the slow fixed BM=128/BN=128 config while the fused kernel "
            "uses its own tiling — this inflates the speedup ~2.4x (a measurement artifact, "
            "not a real win). Run correctness() and bench_real() in SEPARATE processes.")
    torch.manual_seed(0)
    b, d = 1, 128
    scale = 1.0 / _m.sqrt(d)
    nqb = s // block_size
    lab = torch.load(label_path, map_location='cpu', weights_only=False).bool()  # (L, Hkv)
    per_layer = lab.to(torch.int).sum(1).tolist()
    counts = Counter(n for n in per_layer if n > 0)
    total_anchor_hl = int(lab.sum().item())
    print(f"\n[REAL anchor-path s={s//1024}K  label={label_path.split('/')[-2][:40]}...]")
    print(f"  anchor head-layers total = {total_anchor_hl}, per-nf_kv layer counts = {dict(sorted(counts.items()))}")

    tot_p = tot_f = 0.0
    for nf_kv in sorted(counts):
        nfq = nf_kv * G
        chunk_qb = min(nqb, max(1, _m.ceil(chunk_ctas / (nfq * b))))
        q = torch.randn(b, nfq, s, d, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(b, nf_kv, s, d, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(b, nf_kv, s, d, device=DEV, dtype=torch.bfloat16)
        full_bm = torch.ones((b, nfq, nqb, nqb), dtype=torch.bool, device=DEV)
        tp = timed(lambda: block_sparse_attn_with_score(
            q, k, v, full_bm, block_size=block_size, segment_size=segment_size,
            causal=True, softmax_scale=scale, q_chunk_blocks=chunk_qb))
        tf = timed(lambda: gqa_fused_with_score(
            q, k, v, block_size=block_size, softmax_scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, q_chunk_blocks=chunk_qb))
        nl = counts[nf_kv]
        tot_p += tp * nl
        tot_f += tf * nl
        print(f"  nf_kv={nf_kv} (x{nl} layers, chunk_qb={chunk_qb}): "
              f"prod {tp:.3f}ms  fused {tf:.3f}ms  ({tp/tf:.2f}x)")
    print(f"  === anchor-path TOTAL: prod {tot_p:.1f}ms  fused {tot_f:.1f}ms  "
          f"=> {tp/tf if False else tot_p/tot_f:.2f}x, saved {tot_p-tot_f:.1f}ms ===")


if __name__ == '__main__':
    # Two modes, MUST be run in separate processes (the autotune flag is read once
    # at import time in Reuse_v1):
    #   correctness (bit-identical): BLOCK_SPARSE_NO_AUTOTUNE=1 python ... --check
    #   speed (fair, autotune ON):   python ... --bench
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "--bench"
    LABEL = ("attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/"
             "hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-"
             "multi_passkey10-sp8-mb8-xb64/label.pt")
    if mode == "--check":
        ok = correctness()
        print("CORRECTNESS:", "PASS (bit-identical)" if ok else "FAIL")
    else:
        bench_real(LABEL, s=128 * 1024)
        bench_real(LABEL, s=32 * 1024)

