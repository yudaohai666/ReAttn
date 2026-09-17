"""Per-head equivalence check for the batched-anchor reuse_v1_layer_per_hkv.

Reference = the ORIGINAL per-kv-head anchor loop (one block_sparse_attn_with_score
launch per head, q_chunk_blocks=None) + the original selection. Candidate = the new
batched-anchor path inside reuse_v1_layer_per_hkv (all heads anchor, so only the
anchor path runs). We compare, per kv-head:
  - dense output `out`   -> max abs diff (kernel-level; may be ~0 or tiny FP noise
                            if the autotuner picks a different config for the larger
                            batched H bucket)
  - selection k_sel/k_cnt -> exact equality (this is what sparse heads reuse; if it
                            matches, the method's downstream behaviour is unchanged)
"""
import os, sys
import torch
from pbs_attn.baselines.Reuse_v1 import (
    block_sparse_attn_with_score, select_topk_blocks, _select_blocks_topp,
    _compact_block_mask, _resolve_max_sel, reuse_v1_layer_per_hkv, IndexCache,
)

DEV = 'cuda'


def ref_anchor(q, k, v, budget, block_size, segment_size, causal, scale,
               sink_blocks, local_blocks, select_mode, top_p, min_blocks,
               max_blocks, per_head_topp):
    """Original per-head anchor path. Returns (out, {hkv: (k_sel, k_cnt)})."""
    b, H, s, d = q.shape
    Hkv = k.shape[1]
    G = H // Hkv
    nqb = (s + block_size - 1) // block_size
    nkb = nqb
    qg = q.view(b, Hkv, G, s, d)
    out = torch.empty_like(q).view(b, Hkv, G, s, d)
    full_bm = torch.ones((b, G, nqb, nkb), dtype=torch.bool, device=q.device)
    sel = {}
    for hkv in range(Hkv):
        q_h = qg[:, hkv].contiguous()
        k_h = k[:, hkv:hkv + 1].contiguous()
        v_h = v[:, hkv:hkv + 1].contiguous()
        out_h, bs_h = block_sparse_attn_with_score(
            q_h, k_h, v_h, full_bm, block_size=block_size, segment_size=segment_size,
            causal=causal, softmax_scale=scale, q_chunk_blocks=None)
        out[:, hkv] = out_h
        if select_mode == 'topk':
            max_sel = _resolve_max_sel('topk', budget, max_blocks, nqb,
                                       sink_blocks=sink_blocks, local_blocks=local_blocks)
            mask = select_topk_blocks(bs_h, budget=budget, causal=causal,
                                      force_first=(sink_blocks > 0), agg='max',
                                      sink_blocks=sink_blocks, local_blocks=local_blocks)
            ks, kc = _compact_block_mask(mask, max_sel)
        else:
            max_sel = _resolve_max_sel('topp', budget, max_blocks, nqb,
                                       sink_blocks=sink_blocks, local_blocks=local_blocks)
            if per_head_topp:
                ks, kc = _select_blocks_topp(bs_h.reshape(b * G, nqb, nkb), top_p=top_p,
                                             min_blocks=min_blocks, max_blocks=max_blocks,
                                             max_sel=max_sel, causal=causal, nkb=nkb,
                                             nqb=nqb, dev=q.device)
                ks = ks.reshape(b, G, nqb, max_sel); kc = kc.reshape(b, G, nqb)
            else:
                imp = bs_h.amax(dim=1)
                ks, kc = _select_blocks_topp(imp, top_p=top_p, min_blocks=min_blocks,
                                             max_blocks=max_blocks, max_sel=max_sel,
                                             causal=causal, nkb=nkb, nqb=nqb, dev=q.device)
        sel[hkv] = (ks.clone(), kc.clone())
    return out.reshape(b, H, s, d), sel


def run_case(select_mode, per_head_topp, nqb, force_chunk, seed=0):
    torch.manual_seed(seed)
    b, Hkv, G, d = 1, 8, 4, 128
    H = Hkv * G
    block_size = 128
    s = nqb * block_size
    scale = 1.0 / (d ** 0.5)
    top_p, min_blocks, max_blocks, budget = 0.75, min(16, nqb), 1024, 32
    sink_blocks, local_blocks = 1, 2

    q = torch.randn(b, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, Hkv, s, d, device=DEV, dtype=torch.bfloat16)

    out_ref, sel_ref = ref_anchor(
        q, k, v, budget, block_size, 2048, True, scale, sink_blocks, local_blocks,
        select_mode, top_p, min_blocks, max_blocks, per_head_topp)

    if force_chunk:
        # Force q_chunk_blocks < nqb to exercise the chunked path.
        os.environ['REUSE_V1_ANCHOR_MASS_CAP_BYTES'] = str(b * H * nqb * 4 * (block_size * (nqb // 4 or 1)))
    else:
        os.environ.pop('REUSE_V1_ANCHOR_MASS_CAP_BYTES', None)

    max_sel = _resolve_max_sel(select_mode, budget, max_blocks, nqb,
                               sink_blocks=sink_blocks, local_blocks=local_blocks)
    cache = IndexCache(b, nqb, Hkv, max_sel, DEV, per_head=per_head_topp, G=G)
    label = torch.ones(Hkv, dtype=torch.bool)   # all anchor -> only anchor path
    out_new = reuse_v1_layer_per_hkv(
        q, k, v, label, cache, budget=budget, block_size=block_size, segment_size=2048,
        causal=True, softmax_scale=scale, sink_blocks=sink_blocks, local_blocks=local_blocks,
        select_mode=select_mode, top_p=top_p, min_blocks=min_blocks, max_blocks=max_blocks,
        last_q_full=False, per_head_topp=per_head_topp)

    out_diff = (out_new.float() - out_ref.float()).abs().max().item()

    sel_ok = True
    for hkv in range(Hkv):
        ks_ref, kc_ref = sel_ref[hkv]
        ks_new, kc_new = cache.read(hkv)
        if not torch.equal(kc_ref.to(torch.int32), kc_new.to(torch.int32)):
            sel_ok = False; print(f"  [hkv={hkv}] k_cnt MISMATCH"); break
        if not torch.equal(ks_ref.to(torch.int32), ks_new.to(torch.int32)):
            sel_ok = False
            nd = (ks_ref.to(torch.int32) != ks_new.to(torch.int32)).sum().item()
            print(f"  [hkv={hkv}] k_sel MISMATCH ndiff={nd}"); break
    tag = f"mode={select_mode} per_head={per_head_topp} nqb={nqb} chunk={force_chunk}"
    print(f"{tag}: out_max_abs_diff={out_diff:.3e} selection_equal={sel_ok}")
    return sel_ok


def main():
    all_ok = True
    for mode, ph in [('topp', False), ('topp', True), ('topk', False)]:
        for nqb in (8, 32):
            for chunk in (False, True):
                all_ok &= run_case(mode, ph, nqb, chunk)
    print("ALL SELECTION EQUAL" if all_ok else "SELECTION MISMATCH DETECTED")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
