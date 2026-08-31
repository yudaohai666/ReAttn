"""
Kernel-level profiling for reuse_v1 at 128K.
Usage:
  CUDA_VISIBLE_DEVICES=0 python profile_kernel.py
"""
import os, sys, math, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

REPO = os.path.dirname(os.path.abspath(__file__))
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
TOPK_RATIO = None   # topp mode
SELECT_MODE = 'topp'
TOP_P = 0.9
MIN_BLOCKS = 8
MAX_BLOCKS = 64
SINK  = 1
LOCAL = 2
N_WARMUP = 2
N_ITER   = 5
device   = "cuda"

# ------------------------------------------------------------------ #
# Import internals
# ------------------------------------------------------------------ #
from pbs_attn.baselines.Reuse_v1 import (
    block_sparse_attn_with_score,
    select_topk_blocks,
    _compact_block_mask,
    _block_sparse_indexed_fwd,
    _sparse_block_attn,
    _resolve_max_sel,
    IndexCache,
)
import triton

# ------------------------------------------------------------------ #
# Build fake QKV
# ------------------------------------------------------------------ #
torch.manual_seed(0)
q  = torch.randn(B, H,   SEQ, D, device=device, dtype=torch.bfloat16)
k  = torch.randn(B, Hkv, SEQ, D, device=device, dtype=torch.bfloat16)
v  = torch.randn(B, Hkv, SEQ, D, device=device, dtype=torch.bfloat16)

# Load label — profile a typical mixed layer (1 anchor + 7 sparse)
label = torch.load(LABEL_PATH, map_location=device, weights_only=True)
label = (label.float() > 0.5).bool()   # (32, 8)

# First populate cache using layer 0 (all-anchor)
label_L0 = label[0]   # all True
anchor_kv_L0 = label_L0.nonzero().flatten().tolist()

# Profile layer that has most anchors among non-layer-0 layers
anchor_counts = label[1:].sum(dim=-1)  # (31,)
best_layer = int(anchor_counts.argmax().item()) + 1
label_L = label[best_layer]
print(f"Profiling layer {best_layer}: anchor_kv={label_L.nonzero().flatten().tolist()}, "
      f"sparse_kv={(~label_L).nonzero().flatten().tolist()}")

anchor_kv = label_L.nonzero().flatten().tolist()
sparse_kv  = (~label_L).nonzero().flatten().tolist()

nqb = SEQ // BLOCK
nkb = nqb
softmax_scale = 1.0 / math.sqrt(D)
full_bm = torch.ones((B, G, nqb, nkb), dtype=torch.bool, device=device)

q_grouped   = q.view(B, Hkv, G, SEQ, D)
out         = torch.empty_like(q)
out_grouped = out.view(B, Hkv, G, SEQ, D)

# ------------------------------------------------------------------ #
# CUDA event timer helper
# ------------------------------------------------------------------ #
class CudaTimer:
    def __init__(self, name):
        self.name = name
        self.times = []
        self._start = torch.cuda.Event(enable_timing=True)
        self._end   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self._start.record()
        return self

    def __exit__(self, *_):
        self._end.record()
        torch.cuda.synchronize()
        self.times.append(self._start.elapsed_time(self._end))

timers = collections.defaultdict(list)  # name -> [ms]

def record(name, fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    timers[name].append(start.elapsed_time(end))
    return out

# ------------------------------------------------------------------ #
# Pre-run anchor pass once to populate cache (needed for sparse pass)
# ------------------------------------------------------------------ #
max_sel = _resolve_max_sel(SELECT_MODE, 32, MAX_BLOCKS, nqb,
                           sink_blocks=SINK, local_blocks=LOCAL,
                           topk_ratio=TOPK_RATIO)
cache = IndexCache(B, nqb, Hkv, max_sel, device)

# ---- anchor heads (hkv loop) ----
print(f"anchor_kv={anchor_kv}, sparse_kv={sparse_kv}")
print(f"nqb={nqb}, max_sel={max_sel}, select_mode={SELECT_MODE}")
if TOPK_RATIO is not None:
    print(f"topk_ratio={TOPK_RATIO} => per-last-qb budget ~{math.ceil(nqb*TOPK_RATIO)+SINK+LOCAL}")
else:
    print(f"topp: top_p={TOP_P}, min_blocks={MIN_BLOCKS}, max_blocks={MAX_BLOCKS}")

# warm up Triton autotune — use all-anchor label_L0 to populate all cache slots
from pbs_attn.baselines.Reuse_v1 import _select_blocks_topp

def _do_select(bs, imp, max_sel):
    if SELECT_MODE == 'topk':
        mask = select_topk_blocks(bs, budget=32, causal=True,
                                  force_first=True, agg='max',
                                  topk_ratio=TOPK_RATIO,
                                  sink_blocks=SINK, local_blocks=LOCAL)
        return _compact_block_mask(mask, max_sel)
    else:
        return _select_blocks_topp(
            imp, top_p=TOP_P, min_blocks=MIN_BLOCKS, max_blocks=MAX_BLOCKS,
            max_sel=max_sel, causal=True, nkb=nkb, nqb=nqb, dev=device)

for _ in range(N_WARMUP):
    for hkv in anchor_kv_L0:
        q_h = q_grouped[:, hkv].contiguous()
        k_h = k[:, hkv:hkv+1].contiguous()
        v_h = v[:, hkv:hkv+1].contiguous()
        out_h, bs = block_sparse_attn_with_score(
            q_h, k_h, v_h, full_bm,
            block_size=BLOCK, causal=True, softmax_scale=softmax_scale)
        imp = bs.amax(dim=1)
        k_sel, k_cnt = _do_select(bs, imp, max_sel)
        cache.write(hkv, k_sel, k_cnt)
    for hkv in range(Hkv):
        q_h = q_grouped[:, hkv].contiguous()
        k_h = k[:, hkv:hkv+1].contiguous()
        v_h = v[:, hkv:hkv+1].contiguous()
        k_sel_h, k_cnt_h = cache.read(hkv)
        last_qb = nqb - 1
        last_q_start = last_qb * BLOCK
        _sparse_block_attn(
            q_h[:, :, :last_q_start], k_h, v_h, None, 32, BLOCK, True, softmax_scale,
            k_sel=k_sel_h[:, :last_qb].contiguous(),
            k_cnt=k_cnt_h[:, :last_qb].contiguous(),
            out=out_grouped[:, hkv, :, :last_q_start],
        )
torch.cuda.synchronize()
print("Warmup done.")

# ------------------------------------------------------------------ #
# Profiling runs
# ------------------------------------------------------------------ #
# Breakdown categories
cats = ["anchor_dense", "block_score_amax", "select_topk", "compact_mask",
        "cache_write", "cache_read", "q_slice", "kv_slice",
        "sparse_kernel", "last_q_flash", "out_wb"]

wall_times = []

for it in range(N_ITER):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # --- anchor pass ---
    for hkv in anchor_kv:
        # q/k/v slicing
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        q_h = q_grouped[:, hkv].contiguous()
        k_h = k[:, hkv:hkv+1].contiguous()
        v_h = v[:, hkv:hkv+1].contiguous()
        end.record(); torch.cuda.synchronize()
        timers["q_slice"].append(start.elapsed_time(end))

        # anchor dense
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_h, block_score_h = block_sparse_attn_with_score(
            q_h, k_h, v_h, full_bm,
            block_size=BLOCK, causal=True, softmax_scale=softmax_scale)
        end.record(); torch.cuda.synchronize()
        timers["anchor_dense"].append(start.elapsed_time(end))

        # amax
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        imp = block_score_h.amax(dim=1)
        end.record(); torch.cuda.synchronize()
        timers["block_score_amax"].append(start.elapsed_time(end))

        # select blocks (topk or topp unified via _do_select)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        k_sel, k_cnt = _do_select(block_score_h, imp, max_sel)
        end.record(); torch.cuda.synchronize()
        timers["select_blocks"].append(start.elapsed_time(end))

        # cache write
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        cache.write(hkv, k_sel, k_cnt)
        end.record(); torch.cuda.synchronize()
        timers["cache_write"].append(start.elapsed_time(end))

        # out_wb (write out_h back to grouped)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_grouped[:, hkv] = out_h
        end.record(); torch.cuda.synchronize()
        timers["out_wb"].append(start.elapsed_time(end))

    # --- sparse pass ---
    for hkv in sparse_kv:
        # kv slice
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        q_h = q_grouped[:, hkv].contiguous()
        k_h = k[:, hkv:hkv+1].contiguous()
        v_h = v[:, hkv:hkv+1].contiguous()
        end.record(); torch.cuda.synchronize()
        timers["kv_slice"].append(start.elapsed_time(end))

        # cache read
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        k_sel_h, k_cnt_h = cache.read(hkv)
        end.record(); torch.cuda.synchronize()
        timers["cache_read"].append(start.elapsed_time(end))

        last_qb = nqb - 1
        last_q_start = last_qb * BLOCK

        # sparse kernel (body)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        _sparse_block_attn(
            q_h[:, :, :last_q_start], k_h, v_h,
            None, 32, BLOCK, True, softmax_scale,
            k_sel=k_sel_h[:, :last_qb].contiguous(),
            k_cnt=k_cnt_h[:, :last_qb].contiguous(),
            out=out_grouped[:, hkv, :, :last_q_start],
        )
        end.record(); torch.cuda.synchronize()
        timers["sparse_kernel"].append(start.elapsed_time(end))

        # last q flash
        from flash_attn import flash_attn_func
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        last_q = q_h[:, :, last_q_start:].permute(0,2,1,3)  # (b,sq,G,d)
        k_fa   = k_h[:, 0, :, :].unsqueeze(2)               # (b,s,1,d)
        v_fa   = v_h[:, 0, :, :].unsqueeze(2)
        last_out = flash_attn_func(last_q, k_fa, v_fa, causal=True, softmax_scale=softmax_scale)
        end.record(); torch.cuda.synchronize()
        timers["last_q_flash"].append(start.elapsed_time(end))

        # out wb (last q)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_grouped[:, hkv, :, last_q_start:] = last_out.permute(0,2,1,3)
        end.record(); torch.cuda.synchronize()
        timers["out_wb_last"].append(start.elapsed_time(end))

    torch.cuda.synchronize()
    wall_times.append((time.perf_counter() - t0) * 1000)

# ------------------------------------------------------------------ #
# Report
# ------------------------------------------------------------------ #
wall = sum(wall_times) / len(wall_times)

print(f"\n=== Kernel-level profile (avg over {N_ITER} iters) ===")
print(f"Wall-clock (attn only, no HF): {wall:.1f} ms\n")

fmt = "{:<22} {:>8} {:>8} {:>6}  {:>10}"
print(fmt.format("Op", "GPU_ms", "calls", "%wall", "GPU/call"))
print("-" * 64)

total_gpu = 0
rows = []
for name, ts in sorted(timers.items(), key=lambda x: -sum(x[1])):
    gpu_ms  = sum(ts) / N_ITER
    calls   = len(ts) // N_ITER
    per_call = gpu_ms / max(calls, 1)
    pct      = gpu_ms / wall * 100
    total_gpu += gpu_ms
    rows.append((name, gpu_ms, calls, per_call, pct))

for name, gpu_ms, calls, per_call, pct in rows:
    print(fmt.format(name, f"{gpu_ms:.3f}", calls, f"{pct:.1f}%", f"{per_call:.4f}"))

print("-" * 64)
print(fmt.format("SUM GPU", f"{total_gpu:.3f}", "", "", ""))
print(f"wall (attn)      {wall:.3f}")
print(f"unexplained gap  {wall - total_gpu:.3f}  ← Python loop + launch overhead\n")

# ---- detail for slowest ops ----
for name, gpu_ms, calls, per_call, pct in rows[:3]:
    ts = timers[name]
    per_iter = [sum(ts[i*calls:(i+1)*calls]) for i in range(N_ITER)]
    print(f"--- {name} detail ---")
    print(f"  count={calls}/fwd, mean={per_call:.3f}ms, "
          f"min={min(ts)/1:.3f}ms, max={max(ts)/1:.3f}ms")
