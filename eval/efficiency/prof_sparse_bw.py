"""Profile: is reuse_v1's sparse-indexed path KV-bandwidth-bound at 128K?

The GQA-fusion trick (load each selected K/V block ONCE for all G q-heads instead
of G times) only pays off if the kernel is KV-BW-bound. This measures the real
production `_block_sparse_indexed_fwd` (grid over H, per-head K/V reload) on a single
kv-group (Hkv=1, G=4) at 128K with a realistic head-shared selection, then computes:
  - achieved HBM bandwidth vs H800 HBM3 peak (~3.35 TB/s)
  - compute-bound floor (bf16)
If achieved BW is a large fraction of peak, fusion (~G x KV traffic cut) has headroom.

Selection model: nohead topp with max_blocks cap; per q-block cnt = min(qb+1, cap)
(causal growth, capped). Sweep cap in {32, 64} to bound the answer. head-shared k_sel
(b, nqb, max_sel) with stride_h=0 broadcast — the exact nohead layout.
"""
import os, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

DEV = "cuda"
H800_PEAK_TBS = 3.35  # HBM3 peak bandwidth, TB/s
H800_BF16_TFLOPS = 989.0  # dense tensor-core bf16 peak (SXM); triton effective is lower


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


def build_selection(nqb, cap, max_sel, device):
    k_cnt = torch.empty((1, nqb), dtype=torch.int32, device=device)
    k_sel = torch.zeros((1, nqb, max_sel), dtype=torch.int32, device=device)
    for qb in range(nqb):
        cnt = min(qb + 1, cap)
        k_cnt[0, qb] = cnt
        # causal-valid block ids: pick the first `cnt` blocks 0..cnt-1 (all <= qb)
        k_sel[0, qb, :cnt] = torch.arange(cnt, dtype=torch.int32, device=device)
    return k_sel, k_cnt


def main():
    from pbs_attn.baselines.Reuse_v1 import _sparse_block_attn
    b, Hkv, G, d = 1, 1, 4, 128
    H = Hkv * G
    s = 128 * 1024
    block_size = 128
    nqb = s // block_size
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(b, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)

    print(f"[sparse-path roofline  s={s//1024}K  Hkv={Hkv} G={G} H={H}  block_size={block_size}]")
    print(f"  H800 HBM3 peak ~{H800_PEAK_TBS} TB/s ; bf16 TC peak ~{H800_BF16_TFLOPS} TFLOP/s\n")

    for cap in (32, 64):
        max_sel = cap + 2
        k_sel, k_cnt = build_selection(nqb, cap, max_sel, DEV)
        total_blocks = int(k_cnt.sum().item())        # per single head
        mean_cnt = total_blocks / nqb

        t_ms = timed(lambda: _sparse_block_attn(
            q, k, v, block_mask=None, budget=cap, block_size=block_size,
            causal=True, softmax_scale=scale, k_sel=k_sel, k_cnt=k_cnt))
        t_s = t_ms / 1000.0

        # KV bytes moved by the PRODUCTION kernel: grid over H => each of G heads
        # reloads its selected K+V blocks independently.
        blocks_loaded = G * total_blocks
        bytes_per_block = 2 * block_size * d * 2  # K+V, bf16
        kv_bytes = blocks_loaded * bytes_per_block
        achieved_tbs = kv_bytes / t_s / 1e12

        # Compute: per (head, q-block, sel k-block): QK + PV = 2 matmuls of
        # (block_size x block_size x d). FLOPs = 2*MACs.
        macs = G * total_blocks * 2 * (block_size * block_size * d)
        flops = 2 * macs
        compute_floor_s = flops / (H800_BF16_TFLOPS * 1e12)
        bw_floor_s = kv_bytes / (H800_PEAK_TBS * 1e12)
        # After fusion: KV traffic /G, compute unchanged.
        bw_floor_fused_s = (kv_bytes / G) / (H800_PEAK_TBS * 1e12)

        print(f"  cap={cap}  mean_cnt={mean_cnt:.1f}  total_sel_blocks/head={total_blocks}")
        print(f"    measured           : {t_ms:.2f} ms")
        print(f"    KV bytes moved      : {kv_bytes/1e9:.2f} GB  -> achieved BW = {achieved_tbs:.2f} TB/s "
              f"({100*achieved_tbs/H800_PEAK_TBS:.0f}% of peak)")
        print(f"    BW-bound floor      : {bw_floor_s*1000:.2f} ms   (KV-load @ peak)")
        print(f"    compute-bound floor : {compute_floor_s*1000:.2f} ms   (QK+PV @ bf16 peak)")
        print(f"    => bottleneck       : {'KV-BW' if bw_floor_s > compute_floor_s else 'COMPUTE'}")
        print(f"    fused BW floor (/G) : {bw_floor_fused_s*1000:.2f} ms  "
              f"-> fusion helps only down to max(fused-BW, compute) = "
              f"{max(bw_floor_fused_s, compute_floor_s)*1000:.2f} ms")
        # crude realized headroom: how much of measured is plausibly KV-load
        print(f"    measured/compute-floor = {t_s/compute_floor_s:.1f}x, "
              f"measured/BW-floor = {t_s/bw_floor_s:.1f}x\n")


if __name__ == "__main__":
    main()
