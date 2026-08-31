"""Trainable (fwd + bwd) block-sparse indexed attention for Reuse_v1.

The forward path mirrors ``Reuse_v1._block_sparse_indexed_fwd`` but additionally
emits per-row log-sum-exp (LSE, base-2) so backward can recover softmax
probabilities. Backward implements standard FA2 recompute-P + dO*V^T -> dP,
D = rowsum(dO*O), dS = P*(dP-D), dQ += dS @ K * scale, dK += dS^T @ Q * scale,
dV += P^T @ dO. dQ has no cross-program conflicts (each Q block writes disjoint
rows). dK/dV use atomic-add into fp32 buffers (parallelized over Q blocks, same
loop shape as forward). GQA is handled by ``pid_h // num_kv_groups`` -> shared
KV head; multiple Q heads accumulate into the same KV head naturally via the
atomic-add.

Selection tensors ``(K_sel, K_cnt)`` are per (batch, q_block, kv_head), exactly
like ``Reuse_v1._compact_block_mask_per_hkv`` / ``IndexCache``. The G q-heads of
a group share their kv-head's slot via ``pid_h // num_kv_groups``. Backward
treats them as constants; mask is not differentiable.
"""

import math
import os
import torch
import triton
import triton.language as tl

# ============================================================================ #
# LSE-emitting forward: same math as Reuse_v1._block_sparse_indexed_fwd, plus
# stores LSE_log2[b, h, m] = m_i + log2(l_i). Kept as a self-contained kernel
# (no autotune) to keep MAX_SEL and LOGICAL_BLOCK_SIZE stable across calls.
# ============================================================================ #
_BWD_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 2}),
]


def _prune_bwd(configs, named_args, **kwargs):
    lbs = kwargs.get('LOGICAL_BLOCK_SIZE', None) or named_args.get('LOGICAL_BLOCK_SIZE', None)
    try:
        lbs = int(lbs)
    except Exception:
        return configs
    return [c for c in configs
            if (lbs % c.kwargs['BLOCK_M']) == 0 and (lbs % c.kwargs['BLOCK_N']) == 0]


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=['H', 'HEAD_DIM', 'LOGICAL_BLOCK_SIZE', 'num_kv_groups'],
    prune_configs_by={'early_config_prune': _prune_bwd},
)
@triton.jit
def _indexed_fwd_lse_kernel(
    Q, K, V, O, LSE,
    K_sel, K_cnt,
    sq_bz, sq_h, sq_m, sq_d,
    sk_bz, sk_h, sk_n, sk_d,
    sv_bz, sv_h, sv_n, sv_d,
    so_bz, so_h, so_m, so_d,
    slse_bz, slse_h, slse_m,
    sksel_z, sksel_q, sksel_hkv, sksel_s,
    skcnt_z, skcnt_q, skcnt_hkv,
    qo_len, kv_len, softmax_scale,
    H: tl.constexpr, num_kv_groups: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr, MAX_SEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_M) == 0)
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_N) == 0)
    NSUB: tl.constexpr = LOGICAL_BLOCK_SIZE // BLOCK_N
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)
    dtype = Q.type.element_ty

    logical_q_block = (pid_seq * BLOCK_M) // LOGICAL_BLOCK_SIZE
    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m < qo_len

    q_base = Q + pid_bz * sq_bz + pid_h * sq_h
    q = tl.load(q_base + offs_m[:, None] * sq_m + offs_d[None, :] * sq_d,
                mask=q_mask[:, None], other=0.0).to(dtype)
    k_base = K + pid_bz * sk_bz + (pid_h // num_kv_groups) * sk_h
    v_base = V + pid_bz * sv_bz + (pid_h // num_kv_groups) * sv_h

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    scale = softmax_scale * 1.44269504

    cnt = tl.load(K_cnt + pid_bz * skcnt_z + logical_q_block * skcnt_q
                  + (pid_h // num_kv_groups) * skcnt_hkv)
    sel_base = (K_sel + pid_bz * sksel_z.to(tl.int64) + logical_q_block * sksel_q
                + (pid_h // num_kv_groups) * sksel_hkv)
    for i in range(0, MAX_SEL):
        if i < cnt:
            kb = tl.load(sel_base + i * sksel_s)
            for sub in tl.static_range(NSUB):
                kv_seq_start = kb * LOGICAL_BLOCK_SIZE + sub * BLOCK_N
                offs_n = kv_seq_start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < kv_len
                k = tl.load(k_base + offs_n[:, None] * sk_n + offs_d[None, :] * sk_d,
                            mask=n_mask[:, None], other=0.0).to(dtype)
                qk = tl.dot(q, tl.trans(k)) * scale
                bad = offs_n[None, :] >= kv_len
                if IS_CAUSAL:
                    bad |= offs_m[:, None] < offs_n[None, :]
                qk = qk + tl.where(bad, -1.0e6, 0.0)
                local_m = tl.max(qk, 1)
                m_ij = tl.maximum(m_i, local_m)
                qk -= m_ij[:, None]
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(m_i - m_ij)
                acc = acc * alpha[:, None]
                v = tl.load(v_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
                            mask=n_mask[:, None], other=0.0).to(dtype)
                acc += tl.dot(p.to(dtype), v)
                l_i = l_i * alpha + l_ij
                m_i = m_ij

    acc = acc / l_i[:, None]
    o_base = O + pid_bz * so_bz + pid_h * so_h
    tl.store(o_base + offs_m[:, None] * so_m + offs_d[None, :] * so_d,
             acc.to(dtype), mask=q_mask[:, None])
    # LSE in log2 basis (matches base of scale factor above).
    lse_log2 = m_i + tl.math.log2(l_i)
    tl.store(LSE + pid_bz * slse_bz + pid_h * slse_h + offs_m * slse_m,
             lse_log2, mask=q_mask)


# ============================================================================ #
# Backward preprocess: D[b,h,m] = rowsum(O[b,h,m,:] * dO[b,h,m,:])
# ============================================================================ #
@triton.jit
def _bwd_preprocess_kernel(
    O, dO, D,
    so_bz, so_h, so_m, so_d,
    sdo_bz, sdo_h, sdo_m, sdo_d,
    sd_bz, sd_h, sd_m,
    qo_len,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < qo_len
    o = tl.load(O + pid_bz * so_bz + pid_h * so_h
                + offs_m[:, None] * so_m + offs_d[None, :] * so_d,
                mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(dO + pid_bz * sdo_bz + pid_h * sdo_h
                 + offs_m[:, None] * sdo_m + offs_d[None, :] * sdo_d,
                 mask=mask_m[:, None], other=0.0).to(tl.float32)
    d = tl.sum(o * do, axis=1)
    tl.store(D + pid_bz * sd_bz + pid_h * sd_h + offs_m * sd_m, d, mask=mask_m)


# ============================================================================ #
# Backward dQ kernel: parallel over (b, h, qb). Iterates the same K_sel list as
# forward. No cross-program conflict (each pid writes a disjoint Q slice), so no
# atomics are needed. Recomputes P via LSE, then dQ += (P*(dP-D)) @ K * scale.
# ============================================================================ #
@triton.autotune(
    configs=_BWD_CONFIGS,
    key=['H', 'HEAD_DIM', 'LOGICAL_BLOCK_SIZE', 'num_kv_groups'],
    prune_configs_by={'early_config_prune': _prune_bwd},
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dO, dQ, LSE, D,
    K_sel, K_cnt,
    sq_bz, sq_h, sq_m, sq_d,
    sk_bz, sk_h, sk_n, sk_d,
    sv_bz, sv_h, sv_n, sv_d,
    sdo_bz, sdo_h, sdo_m, sdo_d,
    sdq_bz, sdq_h, sdq_m, sdq_d,
    slse_bz, slse_h, slse_m,
    sd_bz, sd_h, sd_m,
    sksel_z, sksel_q, sksel_hkv, sksel_s,
    skcnt_z, skcnt_q, skcnt_hkv,
    qo_len, kv_len, softmax_scale,
    H: tl.constexpr, num_kv_groups: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr, MAX_SEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_M) == 0)
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_N) == 0)
    NSUB: tl.constexpr = LOGICAL_BLOCK_SIZE // BLOCK_N
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)
    dtype = Q.type.element_ty

    logical_q_block = (pid_seq * BLOCK_M) // LOGICAL_BLOCK_SIZE
    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m < qo_len
    q_base = Q + pid_bz * sq_bz + pid_h * sq_h
    q = tl.load(q_base + offs_m[:, None] * sq_m + offs_d[None, :] * sq_d,
                mask=q_mask[:, None], other=0.0).to(dtype)
    do_base = dO + pid_bz * sdo_bz + pid_h * sdo_h
    do = tl.load(do_base + offs_m[:, None] * sdo_m + offs_d[None, :] * sdo_d,
                 mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE + pid_bz * slse_bz + pid_h * slse_h + offs_m * slse_m,
                  mask=q_mask, other=0.0)
    D_row = tl.load(D + pid_bz * sd_bz + pid_h * sd_h + offs_m * sd_m,
                    mask=q_mask, other=0.0)

    k_base = K + pid_bz * sk_bz + (pid_h // num_kv_groups) * sk_h
    v_base = V + pid_bz * sv_bz + (pid_h // num_kv_groups) * sv_h
    scale_log2 = softmax_scale * 1.44269504

    dq_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    cnt = tl.load(K_cnt + pid_bz * skcnt_z + logical_q_block * skcnt_q
                  + (pid_h // num_kv_groups) * skcnt_hkv)
    sel_base = (K_sel + pid_bz * sksel_z.to(tl.int64) + logical_q_block * sksel_q
                + (pid_h // num_kv_groups) * sksel_hkv)

    for i in range(0, MAX_SEL):
        if i < cnt:
            kb = tl.load(sel_base + i * sksel_s)
            for sub in tl.static_range(NSUB):
                kv_seq_start = kb * LOGICAL_BLOCK_SIZE + sub * BLOCK_N
                offs_n = kv_seq_start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < kv_len
                k = tl.load(k_base + offs_n[:, None] * sk_n + offs_d[None, :] * sk_d,
                            mask=n_mask[:, None], other=0.0).to(dtype)
                v = tl.load(v_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
                            mask=n_mask[:, None], other=0.0)
                qk = tl.dot(q, tl.trans(k))
                qk_log2 = qk * scale_log2
                bad = offs_n[None, :] >= kv_len
                if IS_CAUSAL:
                    bad |= offs_m[:, None] < offs_n[None, :]
                bad |= (~q_mask)[:, None]
                qk_log2 = qk_log2 + tl.where(bad, -1.0e6, 0.0)
                p = tl.math.exp2(qk_log2 - lse[:, None])
                p = tl.where(bad, 0.0, p)
                # dP = dO @ V^T ; use fp32 accumulate
                dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))
                ds = p * (dp - D_row[:, None])
                dq_acc += tl.dot(ds.to(dtype), k) * softmax_scale

    tl.store(dQ + pid_bz * sdq_bz + pid_h * sdq_h
             + offs_m[:, None] * sdq_m + offs_d[None, :] * sdq_d,
             dq_acc.to(dtype), mask=q_mask[:, None])


# ============================================================================ #
# Backward dK/dV kernel: parallel over (b, h, qb) (same iteration as forward and
# dQ), atomically accumulates into fp32 dK/dV[b, hkv, :, :]. GQA: pid_h maps to
# hkv = pid_h // num_kv_groups, so all q-heads in the same group naturally sum
# into the shared kv-head via the atomic-add.
# ============================================================================ #
@triton.autotune(
    configs=_BWD_CONFIGS,
    key=['H', 'HEAD_DIM', 'LOGICAL_BLOCK_SIZE', 'num_kv_groups'],
    prune_configs_by={'early_config_prune': _prune_bwd},
    reset_to_zero=['dK', 'dV'],   # atomic_add outputs: must be zeroed before each benchmark trial
)
@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, dO, dK, dV, LSE, D,
    K_sel, K_cnt,
    sq_bz, sq_h, sq_m, sq_d,
    sk_bz, sk_h, sk_n, sk_d,
    sv_bz, sv_h, sv_n, sv_d,
    sdo_bz, sdo_h, sdo_m, sdo_d,
    sdk_bz, sdk_h, sdk_n, sdk_d,
    sdv_bz, sdv_h, sdv_n, sdv_d,
    slse_bz, slse_h, slse_m,
    sd_bz, sd_h, sd_m,
    sksel_z, sksel_q, sksel_hkv, sksel_s,
    skcnt_z, skcnt_q, skcnt_hkv,
    qo_len, kv_len, softmax_scale,
    H: tl.constexpr, num_kv_groups: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    LOGICAL_BLOCK_SIZE: tl.constexpr, MAX_SEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_M) == 0)
    tl.static_assert((LOGICAL_BLOCK_SIZE % BLOCK_N) == 0)
    NSUB: tl.constexpr = LOGICAL_BLOCK_SIZE // BLOCK_N
    pid_seq = tl.program_id(0)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_bz = tl.program_id(2).to(tl.int64)
    dtype = Q.type.element_ty

    logical_q_block = (pid_seq * BLOCK_M) // LOGICAL_BLOCK_SIZE
    offs_m = pid_seq * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m < qo_len

    q_base = Q + pid_bz * sq_bz + pid_h * sq_h
    q = tl.load(q_base + offs_m[:, None] * sq_m + offs_d[None, :] * sq_d,
                mask=q_mask[:, None], other=0.0).to(dtype)
    do_base = dO + pid_bz * sdo_bz + pid_h * sdo_h
    do = tl.load(do_base + offs_m[:, None] * sdo_m + offs_d[None, :] * sdo_d,
                 mask=q_mask[:, None], other=0.0)
    lse = tl.load(LSE + pid_bz * slse_bz + pid_h * slse_h + offs_m * slse_m,
                  mask=q_mask, other=0.0)
    D_row = tl.load(D + pid_bz * sd_bz + pid_h * sd_h + offs_m * sd_m,
                    mask=q_mask, other=0.0)

    hkv = pid_h // num_kv_groups
    k_base = K + pid_bz * sk_bz + hkv * sk_h
    v_base = V + pid_bz * sv_bz + hkv * sv_h
    dk_base = dK + pid_bz * sdk_bz + hkv * sdk_h
    dv_base = dV + pid_bz * sdv_bz + hkv * sdv_h
    scale_log2 = softmax_scale * 1.44269504
    cnt = tl.load(K_cnt + pid_bz * skcnt_z + logical_q_block * skcnt_q
                  + hkv * skcnt_hkv)
    sel_base = (K_sel + pid_bz * sksel_z.to(tl.int64) + logical_q_block * sksel_q
                + hkv * sksel_hkv)

    for i in range(0, MAX_SEL):
        if i < cnt:
            kb = tl.load(sel_base + i * sksel_s)
            for sub in tl.static_range(NSUB):
                kv_seq_start = kb * LOGICAL_BLOCK_SIZE + sub * BLOCK_N
                offs_n = kv_seq_start + tl.arange(0, BLOCK_N)
                n_mask = offs_n < kv_len
                k = tl.load(k_base + offs_n[:, None] * sk_n + offs_d[None, :] * sk_d,
                            mask=n_mask[:, None], other=0.0).to(dtype)
                v = tl.load(v_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
                            mask=n_mask[:, None], other=0.0)
                qk = tl.dot(q, tl.trans(k))
                qk_log2 = qk * scale_log2
                bad = offs_n[None, :] >= kv_len
                if IS_CAUSAL:
                    bad |= offs_m[:, None] < offs_n[None, :]
                bad |= (~q_mask)[:, None]
                qk_log2 = qk_log2 + tl.where(bad, -1.0e6, 0.0)
                p = tl.math.exp2(qk_log2 - lse[:, None])
                p = tl.where(bad, 0.0, p)
                # dV += P^T @ dO  (contribution from this Q slice into this K slice)
                dv_contrib = tl.dot(tl.trans(p.to(tl.float32)), do.to(tl.float32))
                # dP = dO @ V^T ; dS = P * (dP - D)
                dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))
                ds = p * (dp - D_row[:, None])
                # dK += dS^T @ Q * scale
                dk_contrib = tl.dot(tl.trans(ds.to(tl.float32)),
                                    q.to(tl.float32)) * softmax_scale
                store_mask = n_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
                tl.atomic_add(dk_base + offs_n[:, None] * sdk_n + offs_d[None, :] * sdk_d,
                              dk_contrib, mask=store_mask)
                tl.atomic_add(dv_base + offs_n[:, None] * sdv_n + offs_d[None, :] * sdv_d,
                              dv_contrib, mask=store_mask)


# ============================================================================ #
# Python-side utilities and autograd wrapper
# ============================================================================ #
def _launch_fwd(q, k, v, k_sel, k_cnt, softmax_scale, causal, block_size):
    b, H, s, d = q.shape
    Hkv = k.shape[1]
    num_kv_groups = H // Hkv
    MAX_SEL = k_sel.shape[-1]
    out = torch.empty_like(q)
    lse = torch.empty((b, H, s), dtype=torch.float32, device=q.device)

    def grid(META):
        return (triton.cdiv(s, META['BLOCK_M']), H, b)

    _indexed_fwd_lse_kernel[grid](
        q, k, v, out, lse, k_sel, k_cnt,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        k_sel.stride(0), k_sel.stride(1), k_sel.stride(2), k_sel.stride(3),
        k_cnt.stride(0), k_cnt.stride(1), k_cnt.stride(2),
        s, s, softmax_scale,
        H=H, num_kv_groups=num_kv_groups, HEAD_DIM=d,
        LOGICAL_BLOCK_SIZE=block_size, MAX_SEL=MAX_SEL, IS_CAUSAL=causal,
    )
    return out, lse
def _launch_bwd(q, k, v, out, lse, dout, k_sel, k_cnt, softmax_scale, causal, block_size):
    b, H, s, d = q.shape
    Hkv = k.shape[1]
    num_kv_groups = H // Hkv
    MAX_SEL = k_sel.shape[-1]

    dout = dout.contiguous()
    D = torch.empty((b, H, s), dtype=torch.float32, device=q.device)
    PRE_BM = 128
    _bwd_preprocess_kernel[(triton.cdiv(s, PRE_BM), H, b)](
        out, dout, D,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        D.stride(0), D.stride(1), D.stride(2),
        s, BLOCK_M=PRE_BM, HEAD_DIM=d,
    )

    dq = torch.zeros_like(q)
    dk_acc = torch.zeros((b, Hkv, s, d), dtype=torch.float32, device=q.device)
    dv_acc = torch.zeros((b, Hkv, s, d), dtype=torch.float32, device=q.device)

    def grid(META):
        return (triton.cdiv(s, META['BLOCK_M']), H, b)

    _bwd_dq_kernel[grid](
        q, k, v, dout, dq, lse, D, k_sel, k_cnt,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
        k_sel.stride(0), k_sel.stride(1), k_sel.stride(2), k_sel.stride(3),
        k_cnt.stride(0), k_cnt.stride(1), k_cnt.stride(2),
        s, s, softmax_scale,
        H=H, num_kv_groups=num_kv_groups, HEAD_DIM=d,
        LOGICAL_BLOCK_SIZE=block_size, MAX_SEL=MAX_SEL, IS_CAUSAL=causal,
    )

    _bwd_dkdv_kernel[grid](
        q, k, v, dout, dk_acc, dv_acc, lse, D, k_sel, k_cnt,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        dk_acc.stride(0), dk_acc.stride(1), dk_acc.stride(2), dk_acc.stride(3),
        dv_acc.stride(0), dv_acc.stride(1), dv_acc.stride(2), dv_acc.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
        k_sel.stride(0), k_sel.stride(1), k_sel.stride(2), k_sel.stride(3),
        k_cnt.stride(0), k_cnt.stride(1), k_cnt.stride(2),
        s, s, softmax_scale,
        H=H, num_kv_groups=num_kv_groups, HEAD_DIM=d,
        LOGICAL_BLOCK_SIZE=block_size, MAX_SEL=MAX_SEL, IS_CAUSAL=causal,
    )
    return dq, dk_acc.to(k.dtype), dv_acc.to(v.dtype)
class BlockSparseIndexedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_sel, k_cnt, softmax_scale, causal, block_size):
        out, lse = _launch_fwd(q, k, v, k_sel, k_cnt, softmax_scale, causal, block_size)
        ctx.save_for_backward(q, k, v, out, lse, k_sel, k_cnt)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, k_sel, k_cnt = ctx.saved_tensors
        dq, dk, dv = _launch_bwd(q, k, v, out, lse, dout, k_sel, k_cnt,
                                 ctx.softmax_scale, ctx.causal, ctx.block_size)
        # k_sel, k_cnt, softmax_scale, causal, block_size are all non-differentiable
        return dq, dk, dv, None, None, None, None, None


def sparse_block_attn_trainable_from_cache(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,   # (b, H, s, d) / (b, Hkv, s, d) / (b, Hkv, s, d)
    k_sel: torch.Tensor,   # (b, num_q_blocks, Hkv, max_sel) int32
    k_cnt: torch.Tensor,   # (b, num_q_blocks, Hkv) int32
    budget: int,
    block_size: int = 128,
    causal: bool = True,
    softmax_scale: float = None,
) -> torch.Tensor:
    """Trainable per-kv-head block-sparse attention. ``k_sel``/``k_cnt`` are the
    already-compacted per-kv-head selection (e.g. an ``IndexCache`` slice);
    treated as constants (no gradient flows to them)."""
    from Reuse_v1 import _MAX_SEL_TABLE
    if budget not in _MAX_SEL_TABLE:
        raise ValueError(f"budget must be one of {sorted(_MAX_SEL_TABLE)}, got {budget}")
    b, H, s, d = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    return BlockSparseIndexedFunc.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        k_sel.contiguous(), k_cnt.contiguous(), softmax_scale, causal, block_size,
    )


# ============================================================================ #
# PyTorch reference (used by tests): reconstruct the per-kv-head (b, Hkv, nqb,
# nkb) block mask from (k_sel, k_cnt), materialize a (b, H, s, s) token mask and
# run dense softmax attention. GQA handled by explicit K/V repeat_interleave.
# ============================================================================ #
def _ref_sparse_block_attn_per_hkv(q, k, v, k_sel, k_cnt, block_size, causal, softmax_scale=None):
    b, H, s, d = q.shape
    Hkv = k.shape[1]
    G = H // Hkv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    nqb = (s + block_size - 1) // block_size
    nkb = nqb
    device = q.device

    # k_sel: (b, nqb, Hkv, max_sel); k_cnt: (b, nqb, Hkv)
    block_mask = torch.zeros(b, Hkv, nqb, nkb, dtype=torch.bool, device=device)
    for bi in range(b):
        for h in range(Hkv):
            for qb in range(nqb):
                cnt = int(k_cnt[bi, qb, h].item())
                for i in range(cnt):
                    kb = int(k_sel[bi, qb, h, i].item())
                    if kb < nkb:
                        block_mask[bi, h, qb, kb] = True

    ke = k.repeat_interleave(G, dim=1)
    ve = v.repeat_interleave(G, dim=1)
    qb_idx = (torch.arange(s, device=device) // block_size).clamp_max(nqb - 1)
    kb_idx = (torch.arange(s, device=device) // block_size).clamp_max(nkb - 1)
    tok_mask = block_mask[:, :, qb_idx][:, :, :, kb_idx]  # (b, Hkv, s, s)
    tok_mask = tok_mask.repeat_interleave(G, dim=1)       # (b, H, s, s)
    if causal:
        cm = torch.tril(torch.ones(s, s, dtype=torch.bool, device=device))[None, None]
        tok_mask = tok_mask & cm

    qf = q.float()
    kf = ke.float()
    vf = ve.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale
    scores = scores.masked_fill(~tok_mask, float('-inf'))
    all_masked = ~tok_mask.any(dim=-1, keepdim=True)
    scores = torch.where(all_masked, torch.zeros_like(scores), scores)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(all_masked, torch.zeros_like(probs), probs)
    out = torch.matmul(probs, vf)
    return out.to(q.dtype)


# ============================================================================ #
# Self-test / verification
# ============================================================================ #
def _make_random_selection_per_hkv(b, Hkv, nqb, nkb, budget, causal, device, seed=0):
    """Build a random per-kv-head selection and compact it to kernel layout.

    Returns (k_sel (b,nqb,Hkv,max_sel) int32, k_cnt (b,nqb,Hkv) int32). Forces
    the sink block and diagonal block (matches
    Reuse_v1.select_topk_blocks_per_kv_head layout)."""
    from Reuse_v1 import _compact_block_mask_per_hkv, _max_sel_for_budget
    g = torch.Generator(device='cpu').manual_seed(seed)
    scores = torch.rand(b, Hkv, nqb, nkb, generator=g)
    off = nkb - nqb
    qb = torch.arange(nqb)[:, None]
    kb = torch.arange(nkb)[None, :]
    layout = (kb <= qb + off) if causal else torch.ones(nqb, nkb, dtype=torch.bool)
    scores = scores.masked_fill(~layout[None, None], -1.0)
    k_ = min(budget, nkb)
    _, topi = scores.topk(k_, dim=-1)
    mask = torch.zeros(b, Hkv, nqb, nkb, dtype=torch.bool)
    mask.scatter_(-1, topi, True)
    mask[:, :, :, 0] = True
    diag_k = (torch.arange(nqb) + off).clamp_(max=nkb - 1)
    mask[:, :, torch.arange(nqb), diag_k] = True
    mask &= layout[None, None]
    mask = mask.to(device)
    max_sel = _max_sel_for_budget(budget, nkb)
    return _compact_block_mask_per_hkv(mask, max_sel)


def _fwd_matches_reuse_v1(seed=0):
    """Confirm the LSE-emitting forward matches Reuse_v1._sparse_block_attn_per_hkv
    (the forward-only kernel), exercising the same (k_sel, k_cnt) path."""
    from Reuse_v1 import _sparse_block_attn_per_hkv
    torch.manual_seed(seed)
    b, H, Hkv, s, d = 2, 8, 2, 4096, 64
    bs, budget = 128, 16
    device = 'cuda'
    dtype = torch.float16
    q = torch.randn(b, H, s, d, device=device, dtype=dtype) * 0.1
    k = torch.randn(b, Hkv, s, d, device=device, dtype=dtype) * 0.1
    v = torch.randn(b, Hkv, s, d, device=device, dtype=dtype) * 0.1
    nqb = (s + bs - 1) // bs
    nkb = nqb
    k_sel, k_cnt = _make_random_selection_per_hkv(b, Hkv, nqb, nkb, budget,
                                                  causal=True, device=device, seed=seed)

    out_ref = _sparse_block_attn_per_hkv(q, k, v, k_sel, k_cnt, budget, block_size=bs,
                                         causal=True, softmax_scale=None)
    out_us = sparse_block_attn_trainable_from_cache(q, k, v, k_sel, k_cnt, budget,
                                                    block_size=bs, causal=True, softmax_scale=None)
    err = (out_ref.float() - out_us.float()).abs().max().item()
    print(f"[fwd_matches_reuse_v1] max |out_ref - out_us| = {err:.3e}")
    assert err < 5e-3, f"forward mismatch vs Reuse_v1: {err}"


def _bwd_matches_torch_ref_fp32(seed=0):
    """Tight-tolerance backward check vs PyTorch reference in fp32."""
    torch.manual_seed(seed)
    b, H, Hkv, s, d = 1, 4, 4, 2560, 64
    bs, budget = 128, 16
    device = 'cuda'
    dtype = torch.float32
    q = torch.randn(b, H, s, d, device=device, dtype=dtype, requires_grad=True) * 0.5
    k = torch.randn(b, Hkv, s, d, device=device, dtype=dtype, requires_grad=True) * 0.5
    v = torch.randn(b, Hkv, s, d, device=device, dtype=dtype, requires_grad=True) * 0.5
    q.retain_grad(); k.retain_grad(); v.retain_grad()
    nqb = (s + bs - 1) // bs
    k_sel, k_cnt = _make_random_selection_per_hkv(b, Hkv, nqb, nqb, budget,
                                                  causal=True, device=device, seed=seed)

    q1 = q.detach().clone().requires_grad_(True)
    k1 = k.detach().clone().requires_grad_(True)
    v1 = v.detach().clone().requires_grad_(True)
    out_us = sparse_block_attn_trainable_from_cache(q1, k1, v1, k_sel, k_cnt, budget,
                                                    block_size=bs, causal=True)
    dO = torch.randn_like(out_us)
    out_us.backward(dO)

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_ref = _ref_sparse_block_attn_per_hkv(q2, k2, v2, k_sel, k_cnt, block_size=bs, causal=True)
    out_ref.backward(dO)

    e_out = (out_us - out_ref).abs().max().item()
    e_dq = (q1.grad - q2.grad).abs().max().item()
    e_dk = (k1.grad - k2.grad).abs().max().item()
    e_dv = (v1.grad - v2.grad).abs().max().item()
    print(f"[bwd_fp32] max err  out={e_out:.3e}  dQ={e_dq:.3e}  dK={e_dk:.3e}  dV={e_dv:.3e}")
    tol = 1e-2
    assert e_out < tol, f"fwd fp32 err {e_out}"
    assert e_dq  < tol, f"dQ  fp32 err {e_dq}"
    assert e_dk  < tol, f"dK  fp32 err {e_dk}"
    assert e_dv  < tol, f"dV  fp32 err {e_dv}"


def _bwd_matches_torch_ref_lowp(seed=0, dtype=torch.float16, causal=True, gqa=True):
    """Wider-tolerance backward check for fp16/bf16 with optional GQA / non-causal."""
    torch.manual_seed(seed)
    b, H, s, d = 1, 8, 3072, 64
    Hkv = 2 if gqa else H
    bs, budget = 128, 16
    device = 'cuda'
    q = (torch.randn(b, H, s, d, device=device, dtype=dtype) * 0.3).requires_grad_(True)
    k = (torch.randn(b, Hkv, s, d, device=device, dtype=dtype) * 0.3).requires_grad_(True)
    v = (torch.randn(b, Hkv, s, d, device=device, dtype=dtype) * 0.3).requires_grad_(True)
    nqb = (s + bs - 1) // bs
    k_sel, k_cnt = _make_random_selection_per_hkv(b, Hkv, nqb, nqb, budget,
                                                  causal=causal, device=device, seed=seed)

    out_us = sparse_block_attn_trainable_from_cache(q, k, v, k_sel, k_cnt, budget,
                                                    block_size=bs, causal=causal)
    dO = torch.randn_like(out_us)
    (out_us.float() * dO.float()).sum().backward(retain_graph=False)
    dq_us, dk_us, dv_us = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q2 = q.detach().float().clone().requires_grad_(True)
    k2 = k.detach().float().clone().requires_grad_(True)
    v2 = v.detach().float().clone().requires_grad_(True)
    out_ref = _ref_sparse_block_attn_per_hkv(q2, k2, v2, k_sel, k_cnt, block_size=bs, causal=causal)
    (out_ref * dO.float()).sum().backward()

    def rel(a, b_):
        return (a.float() - b_.float()).abs().max().item() / (b_.float().abs().max().item() + 1e-6)
    e_out = rel(out_us, out_ref)
    e_dq = rel(dq_us, q2.grad)
    e_dk = rel(dk_us, k2.grad)
    e_dv = rel(dv_us, v2.grad)
    tag = f"{dtype}/causal={causal}/GQA={gqa}"
    print(f"[bwd_lowp {tag}] rel err  out={e_out:.3e}  dQ={e_dq:.3e}  dK={e_dk:.3e}  dV={e_dv:.3e}")
    tol_out = 5e-2
    tol_grad = 1e-1
    assert e_out < tol_out, f"[{tag}] fwd rel {e_out}"
    assert e_dq < tol_grad, f"[{tag}] dQ rel {e_dq}"
    assert e_dk < tol_grad, f"[{tag}] dK rel {e_dk}"
    assert e_dv < tol_grad, f"[{tag}] dV rel {e_dv}"


def _bwd_boundary_seq_not_multiple(seed=0):
    """seq_len not a multiple of block_size."""
    torch.manual_seed(seed)
    b, H, Hkv, s, d = 1, 4, 2, 2508, 64
    bs, budget = 128, 16
    device = 'cuda'
    q = (torch.randn(b, H, s, d, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
    k = (torch.randn(b, Hkv, s, d, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
    v = (torch.randn(b, Hkv, s, d, device=device, dtype=torch.float32) * 0.3).requires_grad_(True)
    nqb = (s + bs - 1) // bs
    k_sel, k_cnt = _make_random_selection_per_hkv(b, Hkv, nqb, nqb, budget,
                                                  causal=True, device=device, seed=seed)

    out_us = sparse_block_attn_trainable_from_cache(q, k, v, k_sel, k_cnt, budget,
                                                    block_size=bs, causal=True)
    dO = torch.randn_like(out_us)
    out_us.backward(dO)
    dq_us, dk_us, dv_us = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_ref = _ref_sparse_block_attn_per_hkv(q2, k2, v2, k_sel, k_cnt, block_size=bs, causal=True)
    out_ref.backward(dO)

    e_out = (out_us - out_ref).abs().max().item()
    e_dq = (dq_us - q2.grad).abs().max().item()
    e_dk = (dk_us - k2.grad).abs().max().item()
    e_dv = (dv_us - v2.grad).abs().max().item()
    print(f"[bwd_boundary s={s}] max err out={e_out:.3e}  dQ={e_dq:.3e}  dK={e_dk:.3e}  dV={e_dv:.3e}")
    assert e_out < 1e-2
    assert e_dq  < 1e-2
    assert e_dk  < 1e-2
    assert e_dv  < 1e-2


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=True)
    _fwd_matches_reuse_v1()
    _bwd_matches_torch_ref_fp32()
    _bwd_matches_torch_ref_lowp(dtype=torch.float16, causal=True,  gqa=True)
    _bwd_matches_torch_ref_lowp(dtype=torch.float16, causal=False, gqa=True)
    _bwd_matches_torch_ref_lowp(dtype=torch.bfloat16, causal=True,  gqa=False)
    _bwd_boundary_seq_not_multiple()
    print("ALL TESTS PASSED")
