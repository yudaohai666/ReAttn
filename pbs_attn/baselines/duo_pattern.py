"""DuoAttention retrieval/streaming head-label loading and binarization.

DuoAttention learns a per-(layer, kv_head) score in [0, 1] stored as a TSV
(``full_attention_heads.tsv``, shape ``(num_layers, num_kv_heads)``). A high
score marks a *retrieval / full* head (dense causal attention); a low score
marks a *streaming* head (sink + local window only).

This module mirrors DuoAttention's own ``load_attn_pattern`` /
``sparsify_attention_heads`` (duo_attn/utils.py) so the binarized labels match
the original method, but returns a ``torch.BoolTensor`` for the prefill kernel
(``True`` == full head, ``False`` == streaming head).
"""

import json
import os

import numpy as np
import torch


def load_full_attention_heads(attn_load_dir, filename="full_attention_heads.tsv"):
    """Load the raw (num_layers, num_kv_heads) float score matrix, clipped to [0, 1].

    ``attn_load_dir`` may be a directory containing ``filename`` or a direct
    path to a ``.tsv`` file.
    """
    if os.path.isdir(attn_load_dir):
        path = os.path.join(attn_load_dir, filename)
    else:
        path = attn_load_dir
    scores = np.loadtxt(path, dtype=float, delimiter="\t")
    scores = np.clip(scores, 0.0, 1.0)
    if scores.ndim != 2:
        raise ValueError(
            f"full_attention_heads must be 2-D (num_layers, num_kv_heads), "
            f"got shape {scores.shape} from {path}")
    return scores


def read_duo_config(attn_load_dir, default_sink=128, default_recent=256):
    """Read sink_size / recent_size from a sibling config.json if present.

    Falls back to the provided defaults when the directory or keys are missing.
    """
    if os.path.isdir(attn_load_dir):
        cfg_path = os.path.join(attn_load_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            return int(cfg.get("sink_size", default_sink)), int(
                cfg.get("recent_size", default_recent))
    return int(default_sink), int(default_recent)


def sparsify_full_attention_heads(scores, threshold=None, sparsity=None, seed=42):
    """Binarize the score matrix into a full/streaming bool label.

    Mirrors DuoAttention's ``sparsify_attention_heads``:
      - ``sparsity`` given -> threshold = quantile(scores, sparsity), so a
        fraction ``sparsity`` of heads become streaming (pruned).
      - else ``threshold`` is used directly (config.json default 0.5).
    Returns ``(label_bool (L, Hkv), realized_sparsity)`` where ``True`` == full.
    """
    scores = np.asarray(scores, dtype=float).copy()
    # Small tie-breaking noise (same as DuoAttention), seeded for reproducibility.
    rng = np.random.default_rng(seed)
    scores = scores + rng.uniform(0, 1e-6, scores.shape)

    if sparsity is not None:
        thr = np.quantile(scores, sparsity)
        if sparsity >= 1:
            thr = 2.0   # prune all -> all streaming
        elif sparsity <= 0:
            thr = -1.0  # prune none -> all full
    else:
        if threshold is None:
            raise ValueError("Either threshold or sparsity must be provided")
        thr = float(threshold)

    full = scores >= thr
    realized_sparsity = 1.0 - float(full.mean())
    return torch.from_numpy(full).bool(), realized_sparsity


def load_duo_label(attn_load_dir, threshold=0.5, sparsity=None, device="cuda"):
    """One-call loader: raw TSV -> binarized full/streaming bool label on device.

    Returns ``(label (L, Hkv) bool, sink_size, recent_size, realized_sparsity)``.
    ``label[l, h] == True`` means kv-head ``h`` of layer ``l`` is a full head.
    """
    scores = load_full_attention_heads(attn_load_dir)
    sink_size, recent_size = read_duo_config(attn_load_dir, default_sink=128, default_recent=256)
    label, realized = sparsify_full_attention_heads(
        scores, threshold=threshold, sparsity=sparsity)
    return label.to(device), sink_size, recent_size, realized
