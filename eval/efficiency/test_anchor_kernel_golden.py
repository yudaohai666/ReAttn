"""Golden reference for the anchor kernel body change.

Phase 1 (--save): run the CURRENT block_sparse_attn_with_score on a fixed random
workload and dump (out, block_score) to a .pt file.
Phase 2 (--check): after the kernel-body edit, re-run and compare bit-identically
(torch.equal) against the saved golden. Any mismatch exits non-zero.

Covers the M-relevant shapes: G=4 GQA group, LOGICAL_BLOCK_SIZE=BLOCK=128,
causal, plus a couple of alt seq-lens / head-counts so the autotuner exercises
several tile configs (the change must be a no-op for every chosen config).
"""
import os, sys
import torch
from pbs_attn.baselines.Reuse_v1 import block_sparse_attn_with_score

DEV = 'cuda'
GOLD = '/tmp/anchor_golden.pt'
BLOCK, SEG, D = 128, 2048, 128
SCALE = 1.0 / (D ** 0.5)


def workloads():
    # (b, Hkv, G, seq_blocks)
    return [
        (1, 1, 4, 16),
        (1, 1, 4, 33),   # non-power, exercises tail block
        (1, 2, 4, 24),
        (1, 1, 8, 20),
    ]


def run_one(b, Hkv, G, nqb, seed):
    torch.manual_seed(seed)
    H = Hkv * G
    s = nqb * BLOCK
    q = torch.randn(b, H, s, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, D, device=DEV, dtype=torch.bfloat16)
    bm = torch.ones(b, H, nqb, nqb, dtype=torch.bool, device=DEV)
    out, bs = block_sparse_attn_with_score(
        q, k, v, bm, block_size=BLOCK, segment_size=SEG, causal=True,
        softmax_scale=SCALE, q_chunk_blocks=None)
    return out, bs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    results = []
    for i, (b, Hkv, G, nqb) in enumerate(workloads()):
        out, bs = run_one(b, Hkv, G, nqb, seed=100 + i)
        results.append((out.cpu(), bs.cpu()))
    if mode == '--save':
        torch.save(results, GOLD)
        print(f"saved {len(results)} golden workloads to {GOLD}")
        return
    ref = torch.load(GOLD)
    ok = True
    for i, ((o_new, b_new), (o_ref, b_ref)) in enumerate(zip(results, ref)):
        o_eq = torch.equal(o_new, o_ref)
        b_eq = torch.equal(b_new, b_ref)
        if not (o_eq and b_eq):
            ok = False
            od = (o_new.float() - o_ref.float()).abs().max().item()
            bd = (b_new.float() - b_ref.float()).abs().max().item()
            print(f"  WL{i}: out_equal={o_eq} score_equal={b_eq} "
                  f"out_maxdiff={od:.3e} score_maxdiff={bd:.3e}")
        else:
            print(f"  WL{i}: bit-identical")
    print("ALL BIT-IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
