"""
Full-model kernel-level profiling for reuse_v1 at 128K.
Simulates all 32 layers in sequence (layer 0 all-anchor, layers 1-31 per label).
Usage:
  CUDA_VISIBLE_DEVICES=0 python profile_kernel_full.py
"""
import os, sys, math, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

LABEL_PATH = (
    "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn_h"
    "/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai"
    "/data/Llama-3.1-8B-Instruct"
    "/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8"
    "/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8.pt"
)
SEQ   = 128 * 1024
B, H, Hkv, D = 1, 32, 8, 128
G     = H // Hkv
BLOCK = 128
# ---- config: topk + topk_ratio + last_q_full ----
SELECT_MODE = 'topk'
TOPK_RATIO  = 0.05
BUDGET      = 32
TOP_P       = 0.9
MIN_BLOCKS  = 8
MAX_BLOCKS  = None   # topk mode: not used
SINK        = 1
LOCAL       = 2
LAST_Q_FULL = True
N_WARMUP    = 2
N_ITER      = 3
device      = "cuda"

# ------------------------------------------------------------------ #
from pbs_attn.baselines.Reuse_v1 import (
    block_sparse_attn_with_score,
    select_topk_blocks,
    _compact_block_mask,
    _sparse_block_attn,
    _resolve_max_sel,
    IndexCache,
)
from pbs_attn.baselines.Reuse_v1 import _select_blocks_topp
from flash_attn import flash_attn_func

# ------------------------------------------------------------------ #
torch.manual_seed(0)
q  = torch.randn(B, H,   SEQ, D, device=device, dtype=torch.bfloat16)
k  = torch.randn(B, Hkv, SEQ, D, device=device, dtype=torch.bfloat16)
v  = torch.randn(B, Hkv, SEQ, D, device=device, dtype=torch.bfloat16)

label = torch.load(LABEL_PATH, map_location=device, weights_only=True)
label = (label.float() > 0.5).bool()   # (num_layers, Hkv)
num_layers = label.shape[0]

nqb = SEQ // BLOCK
nkb = nqb
softmax_scale = 1.0 / math.sqrt(D)
full_bm = torch.ones((B, G, nqb, nkb), dtype=torch.bool, device=device)

q_grouped   = q.view(B, Hkv, G, SEQ, D)
out         = torch.empty_like(q)
out_grouped = out.view(B, Hkv, G, SEQ, D)

max_sel = _resolve_max_sel(SELECT_MODE, BUDGET, MAX_BLOCKS, nqb,
                           sink_blocks=SINK, local_blocks=LOCAL,
                           topk_ratio=TOPK_RATIO)

print(f"num_layers={num_layers}, nqb={nqb}, max_sel={max_sel}")
print(f"select_mode={SELECT_MODE}, topk_ratio={TOPK_RATIO}, last_q_full={LAST_Q_FULL}")

# anchor count per layer
anchor_per_layer = label.sum(dim=-1).tolist()
total_anchors = sum(anchor_per_layer)
total_sparse  = num_layers * Hkv - total_anchors
print(f"Total anchor kv-head calls: {total_anchors}, sparse: {total_sparse}")
print(f"Anchor distribution per layer: {[int(x) for x in anchor_per_layer]}")

def _do_select(bs, imp, max_sel):
    if SELECT_MODE == 'topk':
        mask = select_topk_blocks(bs, budget=BUDGET, causal=True,
                                  force_first=True, agg='max',
                                  topk_ratio=TOPK_RATIO,
                                  sink_blocks=SINK, local_blocks=LOCAL)
        return _compact_block_mask(mask, max_sel)
    else:
        return _select_blocks_topp(
            imp, top_p=TOP_P, min_blocks=MIN_BLOCKS, max_blocks=MAX_BLOCKS,
            max_sel=max_sel, causal=True, nkb=nkb, nqb=nqb, dev=device)

# ------------------------------------------------------------------ #
# Warmup: run full model N_WARMUP times
# ------------------------------------------------------------------ #
cache = IndexCache(B, nqb, Hkv, max_sel, device)

def run_one_model_pass(record_timers=False):
    """Simulate one full prefill sweep over all layers."""
    timers_local = collections.defaultdict(list) if record_timers else None

    def _t(name, fn, *args, **kwargs):
        if not record_timers:
            return fn(*args, **kwargs)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = fn(*args, **kwargs)
        e.record()
        torch.cuda.synchronize()
        timers_local[name].append(s.elapsed_time(e))
        return r

    for layer_idx in range(num_layers):
        label_L = label[layer_idx]
        anchor_kv = label_L.nonzero().flatten().tolist()
        sparse_kv  = (~label_L).nonzero().flatten().tolist()

        if layer_idx == 0:
            # rebuild cache at layer 0
            cache.__init__(B, nqb, Hkv, max_sel, device)

        # --- anchor pass ---
        for hkv in anchor_kv:
            q_h = _t("qkv_slice", lambda: (
                q_grouped[:, hkv].contiguous(),
                k[:, hkv:hkv+1].contiguous(),
                v[:, hkv:hkv+1].contiguous()
            ))
            if record_timers:
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                q_h = q_grouped[:, hkv].contiguous()
                k_h = k[:, hkv:hkv+1].contiguous()
                v_h = v[:, hkv:hkv+1].contiguous()
                e.record(); torch.cuda.synchronize()
                timers_local["qkv_slice"].append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                out_h, bs = block_sparse_attn_with_score(
                    q_h, k_h, v_h, full_bm,
                    block_size=BLOCK, causal=True, softmax_scale=softmax_scale)
                e.record(); torch.cuda.synchronize()
                timers_local["anchor_dense"].append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                imp = bs.amax(dim=1)
                k_sel, k_cnt = _do_select(bs, imp, max_sel)
                e.record(); torch.cuda.synchronize()
                timers_local["select_blocks"].append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                cache.write(hkv, k_sel, k_cnt)
                e.record(); torch.cuda.synchronize()
                timers_local["cache_write"].append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                out_grouped[:, hkv] = out_h
                e.record(); torch.cuda.synchronize()
                timers_local["out_wb"].append(s.elapsed_time(e))
            else:
                q_h = q_grouped[:, hkv].contiguous()
                k_h = k[:, hkv:hkv+1].contiguous()
                v_h = v[:, hkv:hkv+1].contiguous()
                out_h, bs = block_sparse_attn_with_score(
                    q_h, k_h, v_h, full_bm,
                    block_size=BLOCK, causal=True, softmax_scale=softmax_scale)
                imp = bs.amax(dim=1)
                k_sel, k_cnt = _do_select(bs, imp, max_sel)
                cache.write(hkv, k_sel, k_cnt)
                out_grouped[:, hkv] = out_h

        # --- sparse pass ---
        for hkv in sparse_kv:
            last_qb      = nqb - 1
            last_q_start = last_qb * BLOCK

            if record_timers:
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                q_h = q_grouped[:, hkv].contiguous()
                k_h = k[:, hkv:hkv+1].contiguous()
                v_h = v[:, hkv:hkv+1].contiguous()
                e.record(); torch.cuda.synchronize()
                timers_local["qkv_slice"].append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                k_sel_h, k_cnt_h = cache.read(hkv)
                e.record(); torch.cuda.synchronize()
                timers_local["cache_read"].append(s.elapsed_time(e))

                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                _sparse_block_attn(
                    q_h[:, :, :last_q_start], k_h, v_h,
                    None, BUDGET, BLOCK, True, softmax_scale,
                    k_sel=k_sel_h[:, :last_qb].contiguous(),
                    k_cnt=k_cnt_h[:, :last_qb].contiguous(),
                    out=out_grouped[:, hkv, :, :last_q_start],
                )
                e.record(); torch.cuda.synchronize()
                timers_local["sparse_kernel"].append(s.elapsed_time(e))

                if LAST_Q_FULL:
                    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    last_q   = q_h[:, :, last_q_start:].permute(0,2,1,3)
                    k_fa     = k_h[:, 0, :, :].unsqueeze(2)
                    v_fa     = v_h[:, 0, :, :].unsqueeze(2)
                    last_out = flash_attn_func(last_q, k_fa, v_fa, causal=True, softmax_scale=softmax_scale)
                    e.record(); torch.cuda.synchronize()
                    timers_local["last_q_flash"].append(s.elapsed_time(e))

                    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    out_grouped[:, hkv, :, last_q_start:] = last_out.permute(0,2,1,3)
                    e.record(); torch.cuda.synchronize()
                    timers_local["out_wb_last"].append(s.elapsed_time(e))
            else:
                q_h = q_grouped[:, hkv].contiguous()
                k_h = k[:, hkv:hkv+1].contiguous()
                v_h = v[:, hkv:hkv+1].contiguous()
                k_sel_h, k_cnt_h = cache.read(hkv)
                _sparse_block_attn(
                    q_h[:, :, :last_q_start], k_h, v_h,
                    None, BUDGET, BLOCK, True, softmax_scale,
                    k_sel=k_sel_h[:, :last_qb].contiguous(),
                    k_cnt=k_cnt_h[:, :last_qb].contiguous(),
                    out=out_grouped[:, hkv, :, :last_q_start],
                )
                if LAST_Q_FULL:
                    last_q   = q_h[:, :, last_q_start:].permute(0,2,1,3)
                    k_fa     = k_h[:, 0, :, :].unsqueeze(2)
                    v_fa     = v_h[:, 0, :, :].unsqueeze(2)
                    last_out = flash_attn_func(last_q, k_fa, v_fa, causal=True, softmax_scale=softmax_scale)
                    out_grouped[:, hkv, :, last_q_start:] = last_out.permute(0,2,1,3)

    return timers_local

# ------------------------------------------------------------------ #
# Warmup
# ------------------------------------------------------------------ #
print(f"\nWarming up ({N_WARMUP} full model passes)...")
for _ in range(N_WARMUP):
    run_one_model_pass(record_timers=False)
torch.cuda.synchronize()
print("Warmup done.")

# ------------------------------------------------------------------ #
# Profiling runs
# ------------------------------------------------------------------ #
all_timers = collections.defaultdict(list)
wall_times  = []

for it in range(N_ITER):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    t_local = run_one_model_pass(record_timers=True)
    torch.cuda.synchronize()
    wall_times.append((time.perf_counter() - t0) * 1000)
    for k_name, vs in t_local.items():
        all_timers[k_name].extend(vs)

# ------------------------------------------------------------------ #
# Report — full model totals
# ------------------------------------------------------------------ #
wall = sum(wall_times) / len(wall_times)
print(f"\n=== Full-model kernel profile (avg over {N_ITER} iters) ===")
print(f"Wall-clock (all layers, no HF overhead): {wall:.1f} ms\n")

fmt = "{:<22} {:>10} {:>8} {:>8}  {:>10}"
print(fmt.format("Op", "GPU_total", "calls", "%wall", "GPU/call"))
print("-" * 68)

total_gpu = 0.0
rows = []
for name, ts in sorted(all_timers.items(), key=lambda x: -sum(x[1])):
    gpu_ms   = sum(ts) / N_ITER
    calls    = len(ts) // N_ITER
    per_call = gpu_ms / max(calls, 1)
    pct      = gpu_ms / wall * 100
    total_gpu += gpu_ms
    rows.append((name, gpu_ms, calls, per_call, pct))

for name, gpu_ms, calls, per_call, pct in rows:
    print(fmt.format(name, f"{gpu_ms:.2f}ms", calls, f"{pct:.1f}%", f"{per_call:.4f}ms"))

print("-" * 68)
print(fmt.format("SUM GPU", f"{total_gpu:.2f}ms", "", "", ""))
print(f"wall (all layers)  {wall:.2f}ms")
print(f"unexplained gap    {wall - total_gpu:.2f}ms  ← Python loop + launch overhead")

# ------------------------------------------------------------------ #
# Per-layer anchor count breakdown
# ------------------------------------------------------------------ #
print(f"\n=== Per-layer anchor/sparse breakdown ===")
print(f"{'layer':>6}  {'anchors':>8}  {'sparse':>8}  {'anchor_dense_est':>18}  {'sparse_kernel_est':>20}")
anchor_dense_per_call = rows[0][3] if rows else 42.0  # from profiling
sparse_kernel_per_call = next((r[3] for r in rows if r[0] == 'sparse_kernel'), 3.0)
last_q_per_call = next((r[3] for r in rows if r[0] == 'last_q_flash'), 0.6)

for li in range(num_layers):
    n_anc = int(label[li].sum().item())
    n_sp  = Hkv - n_anc
    est_anc = n_anc * anchor_dense_per_call
    est_sp  = n_sp * (sparse_kernel_per_call + (last_q_per_call if LAST_Q_FULL else 0))
    print(f"  L{li:02d}   {n_anc:>8}  {n_sp:>8}  {est_anc:>16.1f}ms  {est_sp:>18.1f}ms")

print(f"\nEstimated attn total: {total_anchors * anchor_dense_per_call + total_sparse * (sparse_kernel_per_call + last_q_per_call):.1f}ms")

# ------------------------------------------------------------------ #
# Detail for top-3 ops
# ------------------------------------------------------------------ #
print(f"\n=== Detail for top-3 ops ===")
for name, gpu_ms, calls, per_call, pct in rows[:3]:
    ts = all_timers[name]
    print(f"--- {name} ---")
    print(f"  total/fwd={gpu_ms:.3f}ms, calls/fwd={calls}, per_call={per_call:.4f}ms")
    print(f"  min_single={min(ts):.4f}ms, max_single={max(ts):.4f}ms, p50={sorted(ts)[len(ts)//2]:.4f}ms")
