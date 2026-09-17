"""Numerical-equivalence check for the optimized _select_blocks_topp (B change).

Reimplements the ORIGINAL concat/double-sort/dedup algorithm inline and compares
its (k_sel, k_cnt) against the new scatter+compact implementation on random
block-score tensors across shapes / top_p / min_blocks / max_blocks. Any mismatch
prints and exits non-zero.
"""
import sys, torch
from pbs_attn.baselines.Reuse_v1 import _select_blocks_topp, _resolve_max_sel


def _orig_select_blocks_topp(block_score, top_p, min_blocks, max_blocks,
                             max_sel, causal, nkb, nqb, dev):
    b = block_score.shape[0]
    off = nkb - nqb
    qb_idx = torch.arange(nqb, device=dev)
    causal_valid_k = (qb_idx + off).clamp_(max=nkb - 1) + 1
    if causal:
        col = torch.arange(nkb, device=dev)[None, :]
        causal_mask2d = col < causal_valid_k[:, None]
        imp = block_score.masked_fill(~causal_mask2d[None], -1.0)
    else:
        imp = block_score
    k_cand = min(max_blocks, nkb)
    topk_vals, topk_idx = imp.topk(k_cand, dim=-1, sorted=True)
    cumsum = topk_vals.clamp(min=0.0).cumsum(dim=-1)
    total = imp.clamp(min=0.0).sum(dim=-1, keepdim=True).clamp(min=1e-9)
    keep = (cumsum / total) <= top_p
    keep[..., :min_blocks] = (topk_vals[..., :min_blocks] >= 0.0)
    diag_k = (qb_idx + off).clamp_(max=nkb - 1)
    sink_row = torch.zeros(b, nqb, 1, dtype=torch.int64, device=dev)
    diag_row = diag_k.view(1, nqb, 1).expand(b, -1, -1)
    forced_idx = torch.cat([sink_row, diag_row], dim=-1)
    sel_idx = torch.where(keep, topk_idx, torch.full_like(topk_idx, nkb))
    all_idx = torch.cat([sel_idx, forced_idx], dim=-1)
    sorted_idx, _ = all_idx.sort(dim=-1)
    shifted = torch.cat([
        torch.full((b, nqb, 1), -1, dtype=sorted_idx.dtype, device=dev),
        sorted_idx[..., :-1],
    ], dim=-1)
    unique_mask = (sorted_idx != shifted) & (sorted_idx < nkb)
    k_cnt = unique_mask.sum(-1).clamp_(max=max_sel).to(torch.int32).contiguous()
    packed = torch.where(unique_mask, sorted_idx, torch.full_like(sorted_idx, nkb))
    packed, _ = packed.sort(dim=-1)
    k_sel = packed[..., :max_sel].to(torch.int32).contiguous()
    return k_sel, k_cnt


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    cases = 0
    for nqb in (4, 32, 256, 1024):
        nkb = nqb
        for top_p in (0.5, 0.75, 0.9):
            for min_blocks in (1, 8, 16):
                for max_blocks in (16, 64, 1024):
                    for causal in (True, False):
                        b = 2
                        # Mix scales / a zero row / ties to stress dedup + nucleus edges.
                        bs = torch.randn(b, nqb, nkb, device=dev).abs()
                        bs[0, 0] = 0.0
                        if nqb >= 2:
                            bs[1, 1, :] = 1.0  # ties
                        max_sel = _resolve_max_sel('topp', 32, max_blocks, nqb,
                                                   sink_blocks=1, local_blocks=2)
                        a_sel, a_cnt = _orig_select_blocks_topp(
                            bs.clone(), top_p, min_blocks, max_blocks, max_sel,
                            causal, nkb, nqb, dev)
                        n_sel, n_cnt = _select_blocks_topp(
                            bs.clone(), top_p=top_p, min_blocks=min_blocks,
                            max_blocks=max_blocks, max_sel=max_sel, causal=causal,
                            nkb=nkb, nqb=nqb, dev=dev)
                        if not torch.equal(a_cnt, n_cnt):
                            print(f"CNT MISMATCH nqb={nqb} top_p={top_p} "
                                  f"min={min_blocks} max={max_blocks} causal={causal}")
                            print("orig", a_cnt.flatten()[:20])
                            print("new ", n_cnt.flatten()[:20])
                            sys.exit(1)
                        if not torch.equal(a_sel, n_sel):
                            diff = (a_sel != n_sel).nonzero()
                            print(f"SEL MISMATCH nqb={nqb} top_p={top_p} "
                                  f"min={min_blocks} max={max_blocks} causal={causal} "
                                  f"ndiff={diff.shape[0]}")
                            p = tuple(diff[0].tolist())
                            print("at", p, "orig", a_sel[p[:-1]][:12], "new", n_sel[p[:-1]][:12])
                            sys.exit(1)
                        cases += 1
    print(f"OK: {cases} cases identical (k_sel and k_cnt byte-equal) on {dev}")


if __name__ == '__main__':
    main()
