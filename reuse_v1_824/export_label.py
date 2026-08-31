"""Derive a reuse_v1 bool label from trained raw gates at a chosen sparsity.

DuoAttention-style: training saves the continuous per-(layer, kv-head) gates
(``anchor_heads*.tsv``); the binary label is a *snapshot* taken at some
threshold. This offline tool lets you pick the operating point AFTER training,
either by an explicit ``--threshold`` or by a target ``--sparsity`` (the desired
fraction of *all* kv-heads -- INCLUDING the forced-anchor layer 0 -- routed to
the cheap block-sparse branch). Since layer 0 can never be sparse, the pool
(layers 1..L-1) is thresholded at the ``sparsity * N / P`` quantile so the
realised global sparse fraction matches the request (up to discretisation).

Gate polarity: g -> 1 = anchor (dense, expensive), g -> 0 = sparse (cheap).
``label[L,h] = True`` means anchor. Layer 0 is FORCED all-anchor (the reuse_v1
inference constraint ``label[0].all()``), so it can never be routed sparse and
is excluded from the quantile that sets the threshold.

The output ``label.pt`` is a bool ``(num_layers, Hkv)`` tensor consumed as-is by
``reuse_prefill.ReuseV1Holder.load_label`` (which re-thresholds at 0.5) -- so the
inference / eval path needs NO changes.

Usage:
    # by target sparsity (fraction of heads that go sparse/cheap)
    python reuse_v1/export_label.py --input_dir <train_out> --sparsity 0.5
    # or an explicit threshold on the raw gate value
    python reuse_v1/export_label.py --gates <path.tsv> --threshold 0.5 --output label.pt
"""

import argparse
import json
import os

import numpy as np
import torch


def _find_gates(input_dir):
    """Pick the raw-gate tsv from a training output dir (latest preferred)."""
    for name in ("anchor_heads_latest.tsv", "anchor_heads.tsv"):
        p = os.path.join(input_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no anchor_heads_latest.tsv / anchor_heads.tsv under {input_dir}")


def _load_gates(path):
    """Load (num_layers, Hkv) float gates, clipped to [0, 1]."""
    g = np.loadtxt(path, delimiter="\t", dtype=float)
    if g.ndim != 2:
        raise ValueError(f"gates at {path} must be 2-D, got shape {g.shape}")
    return np.clip(g, 0.0, 1.0)


def _resolve_threshold(gates, sparsity, threshold, seed):
    """Return (threshold, gates_for_compare). Layer 0 excluded from quantile.

    A tiny deterministic tie-break noise is added before the quantile (mirrors
    duo's sparsify_attention_heads) so equal gate values split predictably.

    ``sparsity`` is the GLOBAL fraction of *all* kv-heads (including layer 0)
    routed to the cheap sparse branch. Since layer 0 is forced all-anchor and
    can never be sparse, every sparse head must come from the trainable pool
    (layers 1..L-1). To realise a global fraction ``s`` we therefore threshold
    the pool at the ``s * N / P`` quantile, where ``N`` is the total number of
    heads and ``P`` the pool size. This makes the realised global sparsity match
    the requested value (up to head-count discretisation).
    """
    rng = np.random.default_rng(seed)
    noisy = gates + rng.uniform(0.0, 1e-6, size=gates.shape)
    if sparsity is not None:
        if sparsity <= 0:
            return -np.inf, noisy      # nothing sparse -> all anchor
        if sparsity >= 1:
            return np.inf, noisy       # all sparse (except forced layer 0)
        # Quantile over trainable layers only (layer 0 is forced anchor and
        # would otherwise skew the percentile with its frozen 1.0 gates).
        pool = noisy[1:] if noisy.shape[0] > 1 else noisy
        # Convert the desired GLOBAL sparse fraction to the pool quantile: all
        # sparse heads live in the pool, so pool_q = s * N / P (clamped).
        n_total = noisy.size
        n_pool = pool.size
        pool_q = min(1.0, sparsity * n_total / n_pool)
        return float(np.quantile(pool, pool_q)), noisy
    if threshold is None:
        threshold = 0.5
    return float(threshold), noisy


def export_label(gates_path, output, sparsity=None, threshold=None, seed=0):
    gates = _load_gates(gates_path)
    num_layers, hkv = gates.shape
    thr, noisy = _resolve_threshold(gates, sparsity, threshold, seed)

    label = noisy >= thr                       # True = anchor (dense)
    label[0] = True                            # layer 0 forced all-anchor
    label_t = torch.from_numpy(label).bool()

    realized_sparsity = 1.0 - float(label.mean())     # fraction routed sparse
    n_anchor = int(label.sum())

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(label_t, output)

    meta = {
        "source_gates": os.path.abspath(gates_path),
        "num_layers": num_layers,
        "num_key_value_heads": hkv,
        "select_by": "sparsity" if sparsity is not None else "threshold",
        "target_sparsity": sparsity,
        "threshold": thr,
        "realized_anchor_sparsity": realized_sparsity,
        "n_anchor": n_anchor,
        "n_total": int(label.size),
        "layer0_forced_anchor": True,
    }
    meta_path = os.path.splitext(output)[0] + "_config.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    per_layer = label.sum(axis=1)
    print(f"gates      : {gates_path}  shape={num_layers}x{hkv}")
    print(f"threshold  : {thr:.6g}"
          + (f"  (from target sparsity {sparsity})" if sparsity is not None else ""))
    print(f"anchors    : {n_anchor}/{label.size}  "
          f"(realized sparse fraction = {realized_sparsity:.4f})")
    print(f"per-layer anchors (L0 forced {hkv}): {per_layer.tolist()}")
    print(f"layer0 all-anchor: {bool(label[0].all())}")
    print(f"-> {output}")
    print(f"-> {meta_path}")
    return label_t, realized_sparsity


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", help="training output dir (auto-find gates tsv)")
    src.add_argument("--gates", help="explicit path to a raw-gate .tsv")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--sparsity", type=float,
                     help="target GLOBAL fraction of all kv-heads (incl. the "
                          "forced-anchor layer 0) routed to the sparse (cheap) "
                          "branch; the pool is thresholded at the s*N/P quantile")
    sel.add_argument("--threshold", type=float,
                     help="explicit gate threshold (default 0.5 if neither given)")
    ap.add_argument("--output", help="output label.pt path")
    ap.add_argument("--seed", type=int, default=0,
                    help="tie-break noise seed (reproducible quantile split)")
    args = ap.parse_args()

    gates_path = args.gates or _find_gates(args.input_dir)
    if args.output:
        output = args.output
    else:
        base_dir = args.input_dir or os.path.dirname(os.path.abspath(gates_path))
        tag = (f"sparsity{args.sparsity}" if args.sparsity is not None
               else f"thr{args.threshold if args.threshold is not None else 0.5}")
        output = os.path.join(base_dir, f"label_{tag}.pt")

    export_label(gates_path, output, sparsity=args.sparsity,
                 threshold=args.threshold, seed=args.seed)


if __name__ == "__main__":
    main()
