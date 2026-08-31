"""Offline verification for the DuoAttention prefill kernel.

Tier 1: block_streaming_attn_func vs a block-level SDPA reference (sink+local
        causal mask) on a small shape -- validates the .so ABI + H800/SM90.
Tier 2: duo_attention_prefill with sparsity=0 (all full) == flash_attn dense.
Tier 3: mixed sparsity smoke -- output finite, shapes correct.

Run: PYTHONPATH=<repo root> python pbs_attn/baselines/verify_duo.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def block_streaming_mask(s, block_size, sink_block_num, local_block_num, device):
    """(s, s) bool causal mask: query row i attends key j iff j's block is a sink
    block, or within local_block_num blocks of i's block (inclusive), and j<=i."""
    idx = torch.arange(s, device=device)
    qb = (idx // block_size).unsqueeze(1)      # (s,1)
    kb = (idx // block_size).unsqueeze(0)      # (1,s)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
    sink = kb < sink_block_num
    local = (kb <= qb) & (kb > qb - local_block_num)
    return causal & (sink | local)


def tier1_kernel_vs_sdpa():
    print("=== Tier 1: block_streaming_attn_func vs SDPA reference ===")
    from block_sparse_attn import block_streaming_attn_func
    torch.manual_seed(0)
    dev = "cuda"
    b, s, qh, kvh, d = 1, 512, 8, 2, 128
    block_size, sink_bn, local_bn = 128, 1, 2
    G = qh // kvh
    q = torch.randn(b, s, qh, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, s, kvh, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, s, kvh, d, device=dev, dtype=torch.bfloat16)
    sm = 1.0 / math.sqrt(d)

    q_u = q.reshape(b * s, qh, d)
    k_u = k.reshape(b * s, kvh, d)
    v_u = v.reshape(b * s, kvh, d)
    cu = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=dev)
    head_mask_type = torch.full((qh,), -1, dtype=torch.int32, device=dev)
    streaming_info = torch.tensor([sink_bn, local_bn] * qh, dtype=torch.int32, device=dev)
    o = block_streaming_attn_func(
        q_u, k_u, v_u, cu, cu, head_mask_type, streaming_info, s, s,
        p_dropout=0.0, softmax_scale=sm, is_causal=True).reshape(b, s, qh, d)

    # SDPA reference (per q-head, its kv-head expanded).
    mask = block_streaming_mask(s, block_size, sink_bn, local_bn, dev)  # (s,s) bool
    qf = q.permute(0, 2, 1, 3).float()                 # (b,qh,s,d)
    kf = k.permute(0, 2, 1, 3).float().repeat_interleave(G, dim=1)
    vf = v.permute(0, 2, 1, 3).float().repeat_interleave(G, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm
    scores = scores.masked_fill(~mask.view(1, 1, s, s), float("-inf"))
    ref = torch.matmul(torch.softmax(scores, dim=-1), vf).permute(0, 2, 1, 3)  # (b,s,qh,d)

    err = (o.float() - ref).abs().max().item()
    print(f"  max abs err = {err:.4e}  (bf16 tolerance ~1e-2)")
    assert err < 3e-2, f"Tier 1 FAILED: err {err} too large"
    print("  Tier 1 PASSED\n")


def tier2_sparsity0_vs_flash():
    print("=== Tier 2: duo prefill sparsity=0 (all full) == flash_attn ===")
    from flash_attn import flash_attn_func
    from pbs_attn.baselines.DuoAttention import build_duo_holder, duo_attention_prefill
    torch.manual_seed(1)
    dev = "cuda"
    b, H, s, d = 1, 32, 512, 128
    Hkv = 8
    G = H // Hkv
    holder = build_duo_holder("ckp/duo/Llama-3.1-8B-Instruct", sparsity=0.0, device=dev)
    q = torch.randn(b, H, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=dev, dtype=torch.bfloat16)
    out = duo_attention_prefill(q, k, v, num_key_value_groups=G, layer_idx=0, holder=holder)

    kf = k.transpose(1, 2).repeat_interleave(G, dim=2)
    vf = v.transpose(1, 2).repeat_interleave(G, dim=2)
    ref = flash_attn_func(q.transpose(1, 2), kf, vf, causal=True).transpose(1, 2)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"  max abs err = {err:.4e}  (should be ~0, both dense flash)")
    assert err < 1e-3, f"Tier 2 FAILED: err {err}"
    print("  Tier 2 PASSED\n")


def tier3_mixed_smoke():
    print("=== Tier 3: mixed sparsity=0.5 smoke ===")
    from pbs_attn.baselines.DuoAttention import build_duo_holder, duo_attention_prefill
    torch.manual_seed(2)
    dev = "cuda"
    b, H, s, d = 1, 32, 1024, 128
    Hkv = 8
    G = H // Hkv
    holder = build_duo_holder("ckp/duo/Llama-3.1-8B-Instruct", sparsity=0.5, device=dev)
    q = torch.randn(b, H, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=dev, dtype=torch.bfloat16)
    out = duo_attention_prefill(q, k, v, num_key_value_groups=G, layer_idx=5, holder=holder)
    assert out.shape == (b, H, s, d), out.shape
    assert torch.isfinite(out).all(), "non-finite output"
    print(f"  output shape {tuple(out.shape)}, all finite")
    print("  Tier 3 PASSED\n")


if __name__ == "__main__":
    tier1_kernel_vs_sdpa()
    tier2_sparsity0_vs_flash()
    tier3_mixed_smoke()
    print("ALL TIERS PASSED")
