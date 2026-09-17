"""PoC: is the anchor bandwidth-bound on GQA K/V reloads?

The production anchor kernel grids over H q-heads and reloads K/V from HBM once per
q-head (G reloads per kv-head). This PoC implements a GQA-FUSED dense causal kernel:
grid over (q-block, KV-head), load each K/V tile to SRAM ONCE, reuse it across the
G query heads in the group. No scoring — this only tests the K/V-reuse hypothesis.

Compare at 128K against:
  - per-head dense (block_sparse_attn_dense_no_score)  -> current anchor's dense floor
  - flash_attn_func                                    -> external floor (also per-head)
If fused << per-head, the anchor is KV-BW-bound and a fused rewrite is worth it.
"""
import math
import torch
import triton
import triton.language as tl

DEV = 'cuda'


@triton.jit
def _gqa_fused_dense_fwd(
    Q, K, V, O,
    sb_q, sh_q, ss_q, sd_q,
    sb_k, sh_k, ss_k, sd_k,
    sb_v, sh_v, ss_v, sd_v,
    sb_o, sh_o, ss_o, sd_o,
    seqlen, softmax_scale,
    G: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    # One program handles BLOCK_M query positions for ALL G heads of one kv-group.
    # The G heads are stacked in the row dimension (G*BLOCK_M rows); a single K/V
    # tile load is reused by every head (row-wise softmax is independent per row).
    pid_m = tl.program_id(0)
    pid_hkv = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    ROWS: tl.constexpr = G * BLOCK_M
    row = tl.arange(0, ROWS)
    g_of = row // BLOCK_M                      # which head each row belongs to
    m_of = row % BLOCK_M                       # local query index within the block
    pos = pid_m * BLOCK_M + m_of               # absolute query position (same across heads)
    offs_d = tl.arange(0, HEAD_DIM)
    scale = softmax_scale * 1.44269504

    q_row = pid_b * sb_q + (pid_hkv * G + g_of) * sh_q + pos * ss_q
    q = tl.load(Q + q_row[:, None] + offs_d[None, :] * sd_q,
                mask=pos[:, None] < seqlen, other=0.0)             # (ROWS, HEAD_DIM)

    k_base = K + pid_b * sb_k + pid_hkv * sh_k
    v_base = V + pid_b * sb_v + pid_hkv * sh_v

    m_i = tl.zeros((ROWS,), tl.float32) - float("inf")
    l_i = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, HEAD_DIM), tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, seqlen)
    else:
        hi = seqlen

    for kj in range(0, hi, BLOCK_N):
        offs_n = kj + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seqlen
        k = tl.load(k_base + offs_n[:, None] * ss_k + offs_d[None, :] * sd_k,
                    mask=n_mask[:, None], other=0.0)               # loaded ONCE per group
        v = tl.load(v_base + offs_n[:, None] * ss_v + offs_d[None, :] * sd_v,
                    mask=n_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale                        # (ROWS, BLOCK_N)
        bad = offs_n[None, :] >= seqlen
        if IS_CAUSAL:
            bad = bad | (pos[:, None] < offs_n[None, :])
        qk = qk + tl.where(bad, -1e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij

    out = acc / l_i[:, None]
    o_row = pid_b * sb_o + (pid_hkv * G + g_of) * sh_o + pos * ss_o
    tl.store(O + o_row[:, None] + offs_d[None, :] * sd_o,
             out.to(O.type.element_ty), mask=pos[:, None] < seqlen)


def gqa_fused(q, k, v, causal, scale, BLOCK_M=64, BLOCK_N=64):
    b, H, s, d = q.shape
    Hkv = k.shape[1]
    G = H // Hkv
    o = torch.empty_like(q)
    grid = (triton.cdiv(s, BLOCK_M), Hkv, b)
    _gqa_fused_dense_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        s, scale, G=G, HEAD_DIM=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IS_CAUSAL=causal,
        num_warps=8, num_stages=2,
    )
    return o


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


def main():
    from pbs_attn.baselines.Reuse_v1 import block_sparse_attn_dense_no_score
    torch.manual_seed(0)
    b, G, s, d = 1, 4, 128 * 1024, 128
    Hkv = 1
    H = Hkv * G
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)

    # correctness on a small shape
    ss = 1024
    qs = torch.randn(b, H, ss, d, device=DEV, dtype=torch.bfloat16)
    ks = torch.randn(b, Hkv, ss, d, device=DEV, dtype=torch.bfloat16)
    vs = torch.randn(b, Hkv, ss, d, device=DEV, dtype=torch.bfloat16)
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    ref = sdpa(qs.reshape(b * H, ss, d), ks.expand(b, H, ss, d).reshape(b * H, ss, d),
               vs.expand(b, H, ss, d).reshape(b * H, ss, d), scale=scale, is_causal=True
               ).reshape(b, H, ss, d)
    got = gqa_fused(qs, ks, vs, True, scale)
    print("fused vs causal sdpa max_abs_diff =", (got.float() - ref.float()).abs().max().item())

    for bm, bn in [(64, 64), (32, 64), (64, 128), (128, 64)]:
        try:
            t = timed(lambda: gqa_fused(q, k, v, True, scale, BLOCK_M=bm, BLOCK_N=bn))
            print(f"gqa_fused BLOCK_M={bm} BLOCK_N={bn}: {t:.3f} ms")
        except Exception as e:
            print(f"gqa_fused BLOCK_M={bm} BLOCK_N={bn}: FAILED {type(e).__name__}: {str(e)[:80]}")

    tD = timed(lambda: block_sparse_attn_dense_no_score(q, k, v, block_size=128,
               segment_size=2048, causal=True, softmax_scale=scale))
    print(f"per-head dense_no_score (current anchor floor): {tD:.3f} ms")
    try:
        from flash_attn import flash_attn_func
        qf = q.permute(0, 2, 1, 3).contiguous()
        kf = k.permute(0, 2, 1, 3).contiguous()
        vf = v.permute(0, 2, 1, 3).contiguous()
        tC = timed(lambda: flash_attn_func(qf, kf, vf, softmax_scale=scale, causal=True))
        print(f"flash_attn_func (external floor): {tC:.3f} ms")
    except Exception as e:
        print(f"flash unavailable: {type(e).__name__}")


if __name__ == '__main__':
    main()
