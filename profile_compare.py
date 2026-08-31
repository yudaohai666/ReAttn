"""
Compare sparse_kernel latency across 3 selection modes at 128K, all 32 layers.
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
SINK, LOCAL = 1, 2
N_WARMUP, N_ITER = 2, 3
device = "cuda"

from pbs_attn.baselines.Reuse_v1 import (
    block_sparse_attn_with_score, select_topk_blocks,
    _compact_block_mask, _sparse_block_attn, _resolve_max_sel, IndexCache,
)
from pbs_attn.baselines.Reuse_v1 import _select_blocks_topp
from flash_attn import flash_attn_func

torch.manual_seed(0)
q = torch.randn(B, H,   SEQ, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, Hkv, SEQ, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, Hkv, SEQ, D, device=device, dtype=torch.bfloat16)

label = torch.load(LABEL_PATH, map_location=device, weights_only=True)
label = (label.float() > 0.5).bool()
num_layers = label.shape[0]
nqb = SEQ // BLOCK
nkb = nqb
softmax_scale = 1.0 / math.sqrt(D)
full_bm = torch.ones((B, G, nqb, nkb), dtype=torch.bool, device=device)
q_grouped   = q.view(B, Hkv, G, SEQ, D)
out         = torch.empty_like(q)
out_grouped = out.view(B, Hkv, G, SEQ, D)

CONFIGS = [
    dict(name="topk  budget=32       ", select_mode='topk',  budget=32,   topk_ratio=None, top_p=None, min_blocks=None, max_blocks=None),
    dict(name="topk  ratio=0.05      ", select_mode='topk',  budget=32,   topk_ratio=0.05, top_p=None, min_blocks=None, max_blocks=None),
    dict(name="topp  p=0.9 max_b=64  ", select_mode='topp',  budget=32,   topk_ratio=None, top_p=0.9,  min_blocks=8,    max_blocks=64),
]

def make_select_fn(cfg, max_sel):
    sm, ratio, p, minb, maxb = cfg['select_mode'], cfg['topk_ratio'], cfg['top_p'], cfg['min_blocks'], cfg['max_blocks']
    def fn(bs, imp):
        if sm == 'topk':
            mask = select_topk_blocks(bs, budget=cfg['budget'], causal=True,
                                      force_first=True, agg='max',
                                      topk_ratio=ratio, sink_blocks=SINK, local_blocks=LOCAL)
            return _compact_block_mask(mask, max_sel)
        else:
            return _select_blocks_topp(imp, top_p=p, min_blocks=minb, max_blocks=maxb,
                                       max_sel=max_sel, causal=True, nkb=nkb, nqb=nqb, dev=device)
    return fn

def run_full(cfg, record=False):
    max_sel = _resolve_max_sel(cfg['select_mode'], cfg['budget'], cfg['max_blocks'], nqb,
                               sink_blocks=SINK, local_blocks=LOCAL, topk_ratio=cfg['topk_ratio'])
    cache = IndexCache(B, nqb, Hkv, max_sel, device)
    do_select = make_select_fn(cfg, max_sel)
    timers = collections.defaultdict(list)

    for layer_idx in range(num_layers):
        label_L   = label[layer_idx]
        anchor_kv = label_L.nonzero().flatten().tolist()
        sparse_kv = (~label_L).nonzero().flatten().tolist()
        if layer_idx == 0:
            cache.__init__(B, nqb, Hkv, max_sel, device)

        for hkv in anchor_kv:
            q_h = q_grouped[:, hkv].contiguous()
            k_h = k[:, hkv:hkv+1].contiguous()
            v_h = v[:, hkv:hkv+1].contiguous()
            out_h, bs = block_sparse_attn_with_score(q_h, k_h, v_h, full_bm,
                block_size=BLOCK, causal=True, softmax_scale=softmax_scale)
            imp = bs.amax(dim=1)
            k_sel, k_cnt = do_select(bs, imp)
            cache.write(hkv, k_sel, k_cnt)
            out_grouped[:, hkv] = out_h

        for hkv in sparse_kv:
            last_qb      = nqb - 1
            last_q_start = last_qb * BLOCK
            q_h = q_grouped[:, hkv].contiguous()
            k_h = k[:, hkv:hkv+1].contiguous()
            v_h = v[:, hkv:hkv+1].contiguous()
            k_sel_h, k_cnt_h = cache.read(hkv)

            if record:
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
            _sparse_block_attn(
                q_h[:, :, :last_q_start], k_h, v_h, None, cfg['budget'], BLOCK, True, softmax_scale,
                k_sel=k_sel_h[:, :last_qb].contiguous(),
                k_cnt=k_cnt_h[:, :last_qb].contiguous(),
                out=out_grouped[:, hkv, :, :last_q_start],
            )
            if record:
                e.record(); torch.cuda.synchronize()
                timers["sparse_kernel"].append(s.elapsed_time(e))

            last_q   = q_h[:, :, last_q_start:].permute(0,2,1,3)
            k_fa     = k_h[:, 0, :, :].unsqueeze(2)
            v_fa     = v_h[:, 0, :, :].unsqueeze(2)
            last_out = flash_attn_func(last_q, k_fa, v_fa, causal=True, softmax_scale=softmax_scale)
            out_grouped[:, hkv, :, last_q_start:] = last_out.permute(0,2,1,3)

    return timers, max_sel

print("=" * 72)
print(f"{'Config':<26} {'MAX_SEL':>8} {'sparse_total':>14} {'calls':>7} {'per_call':>10} {'wall_attn':>11}")
print("-" * 72)

for cfg in CONFIGS:
    # warmup
    for _ in range(N_WARMUP):
        run_full(cfg, record=False)
    torch.cuda.synchronize()

    # measure
    all_sparse = []
    walls = []
    for _ in range(N_ITER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        timers, max_sel = run_full(cfg, record=True)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1000)
        all_sparse.extend(timers["sparse_kernel"])

    calls     = len(all_sparse) // N_ITER
    total_ms  = sum(all_sparse) / N_ITER
    per_call  = total_ms / max(calls, 1)
    wall_avg  = sum(walls) / N_ITER
    print(f"{cfg['name']:<26} {max_sel:>8} {total_ms:>12.2f}ms {calls:>7} {per_call:>8.4f}ms {wall_avg:>9.1f}ms")

print("=" * 72)
