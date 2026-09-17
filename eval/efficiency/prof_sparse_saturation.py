"""Measure sparse-path grid saturation: per-hkv loop vs A1 batched (one launch).

Grid per sparse launch = (cdiv(s, BLOCK_M), H_of_call, b).
  - loop (production): H_of_call = G, one launch per sparse kv-head  -> N launches
  - batched (A1):      H_of_call = N*G, ONE launch for all sparse kv-heads

Same total selected-block work in both -> the time delta is pure launch/occupancy.
"""
import os, sys, math, torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn")
from pbs_attn.baselines.Reuse_v1 import _sparse_block_attn

DEV = "cuda:0"
SM = 132          # H800
b, G, d, BS = 1, 4, 128, 128
N_SPARSE = 6      # realistic avg sparse kv-heads/layer (Hkv=8, ~1.6 anchor)
MAX_BLOCKS = 48
LENGTHS = [8192, 16384, 32768, 65536, 131072]


def build_sel(nqb, nkb, max_sel, H, per_head, dev):
    # causal cnt per q-block, capped at MAX_BLOCKS; ascending indices.
    cnt = torch.tensor([min(i + 1, MAX_BLOCKS) for i in range(nqb)],
                       dtype=torch.int32, device=dev)
    sel = torch.zeros(nqb, max_sel, dtype=torch.int32, device=dev)
    for i in range(nqb):
        c = int(cnt[i])
        sel[i, :c] = torch.arange(c, device=dev, dtype=torch.int32)
        sel[i, c:] = nkb
    if per_head:
        return (sel.view(1, 1, nqb, max_sel).expand(b, H, nqb, max_sel).contiguous(),
                cnt.view(1, 1, nqb).expand(b, H, nqb).contiguous())
    return (sel.view(1, nqb, max_sel).expand(b, nqb, max_sel).contiguous(),
            cnt.view(1, nqb).expand(b, nqb).contiguous())


def timed(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def run(s):
    nqb = (s + BS - 1) // BS
    nkb = nqb
    max_sel = min(MAX_BLOCKS + 2, nkb)
    scale = 1.0 / math.sqrt(d)

    # loop: N heads, each q=(b,G,s,d) k/v=(b,1,s,d), shared sel dim3
    qL = [torch.randn(b, G, s, d, device=DEV, dtype=torch.bfloat16) for _ in range(N_SPARSE)]
    kL = [torch.randn(b, 1, s, d, device=DEV, dtype=torch.bfloat16) for _ in range(N_SPARSE)]
    vL = [torch.randn(b, 1, s, d, device=DEV, dtype=torch.bfloat16) for _ in range(N_SPARSE)]
    selS, cntS = build_sel(nqb, nkb, max_sel, G, False, DEV)

    def loop_fn():
        for i in range(N_SPARSE):
            _sparse_block_attn(qL[i], kL[i], vL[i], None, MAX_BLOCKS, BS, True, scale,
                               k_sel=selS, k_cnt=cntS)

    # batched: q=(b,N*G,s,d) k/v=(b,N,s,d), per-head sel dim4
    qB = torch.randn(b, N_SPARSE * G, s, d, device=DEV, dtype=torch.bfloat16)
    kB = torch.randn(b, N_SPARSE, s, d, device=DEV, dtype=torch.bfloat16)
    vB = torch.randn(b, N_SPARSE, s, d, device=DEV, dtype=torch.bfloat16)
    selP, cntP = build_sel(nqb, nkb, max_sel, N_SPARSE * G, True, DEV)

    def batch_fn():
        _sparse_block_attn(qB, kB, vB, None, MAX_BLOCKS, BS, True, scale,
                           k_sel=selP, k_cnt=cntP)

    t_loop = timed(loop_fn)
    t_batch = timed(batch_fn)
    return nqb, t_loop, t_batch


print(f"{'len':>6} {'nqb':>5} {'grid/launch(BM64)':>17} {'waves(BM64)':>11} "
      f"{'loop_ms':>9} {'batch_ms':>9} {'speedup':>8}")
for s in LENGTHS:
    nqb, tl, tb = run(s)
    # BLOCK_M=64 => q_blocks = cdiv(s,64); per-launch grid = q_blocks*G
    qb64 = (s + 63) // 64
    grid_launch = qb64 * G
    waves = grid_launch / SM
    print(f"{s//1024:>4}k {nqb:>5} {grid_launch:>17} {waves:>11.2f} "
          f"{tl:>9.3f} {tb:>9.3f} {tl/tb:>7.2f}x")
