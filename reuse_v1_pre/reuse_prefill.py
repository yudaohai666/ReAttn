"""HF prefill-patch adapter for cross-layer per-kv-head block-sparse reuse.

Reuses the validated pure-Triton kernel in ``pbs_attn/baselines/Reuse_v1.py``
(runs on H800 / Hopper). Wires it into the ``patch_type='reuse_v1'`` prefill
path: each layer's kv-heads independently act as anchor (dense + fresh top-k
selection) or sparse (reuse their own most-recent anchor selection via a
persistent ``IndexCache``). Decode (q_len==1) is handled by the standard dense
forward, so this module only sees prefill calls.

One ``generate()`` == one prefill sweep over layers 0..L-1. ``layer_idx==0`` is
the reliable "new prompt" signal used to (re)build the cache; later layers
validate the cache binding. The holder is per-model (captured by the partial
returned from ``get_reuse_v1_prefill``), never a global singleton.
"""

import functools
import math
import os
import sys


class ReuseV1Holder:
    """Per-model state for the reuse_v1 prefill path.

    Static config is set once by ``get_reuse_v1_prefill``; ``cache`` / ``_bound``
    are mutable per-prompt state rebuilt at ``layer_idx==0``.
    """

    def __init__(self, budget=32, block_size=128, segment_size=2048,
                 sink_blocks=1, local_blocks=2, causal=True,
                 select_mode='topk', top_p=0.9, min_blocks=8, max_blocks=64):
        self.budget = budget
        self.block_size = block_size
        self.segment_size = segment_size
        self.sink_blocks = sink_blocks
        self.local_blocks = local_blocks
        self.causal = causal
        # Block-selection mode: 'topk' (fixed budget) or 'topp' (nucleus).
        self.select_mode = select_mode
        self.top_p = top_p
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        # Filled in by get_reuse_v1_prefill.
        self.label = None            # (num_layers, Hkv) bool tensor on device
        self.num_layers = None
        # Mutable per-prompt state.
        self.cache = None
        self._bound = None           # (b, nqb, Hkv) the current cache is bound to

    def reset(self):
        self.cache = None
        self._bound = None

    @staticmethod
    def load_label(path, device):
        """Load an anchor/sparse label matrix (num_layers, Hkv) bool.

        Values are thresholded at 0.5; layer 0 must be all-anchor (all True).
        """
        import torch
        x = torch.load(path, map_location='cpu')
        if not hasattr(x, 'dim'):
            raise ValueError(f"label at {path} is not a tensor: {type(x)}")
        label = (x.float() > 0.5)
        if label.dim() != 2:
            raise ValueError(f"label must be 2-D (num_layers, Hkv), got shape {tuple(label.shape)}")
        if not bool(label[0].all()):
            raise ValueError("layer 0 must be all-anchor (label[0].all() == True)")
        return label.to(device).bool()


def reuse_v1_prefill(query_states, key_states, value_states,
                     num_key_value_groups=None, layer_idx=None, *, holder):
    """One-layer prefill entry for the HF patch.

    ``num_key_value_groups`` / ``layer_idx`` MUST stay explicit named params so
    ``patched_attention_forward`` detects them (via ``__code__.co_varnames``) and
    (a) passes ``layer_idx`` and (b) keeps K/V native GQA instead of repeat_kv'ing
    them. ``num_key_value_groups`` is only used for a sanity check -- it is NOT
    forwarded (``reuse_v1_layer_per_hkv`` has no such param).
    """
    from pbs_attn.baselines.Reuse_v1 import reuse_v1_layer_per_hkv, IndexCache, _resolve_max_sel

    if layer_idx is None:
        raise RuntimeError("reuse_v1_prefill requires layer_idx (patch must pass it)")

    b, H, s, d = query_states.shape
    Hkv = key_states.shape[1]
    G = H // Hkv
    if num_key_value_groups is not None and num_key_value_groups != G:
        raise ValueError(
            f"num_key_value_groups={num_key_value_groups} != H//Hkv={G}; "
            "K/V must be native GQA (not repeat_kv'd) for reuse_v1")
    nqb = (s + holder.block_size - 1) // holder.block_size
    dev = query_states.device

    if layer_idx == 0:
        max_sel = _resolve_max_sel(holder.select_mode, holder.budget, holder.max_blocks, nqb)
        holder.cache = IndexCache(b, nqb, Hkv, max_sel, dev)
        holder._bound = (b, nqb, Hkv)
    else:
        if holder.cache is None:
            raise RuntimeError(
                f"reuse_v1: layer {layer_idx} reached with no cache; layer 0 was "
                "not patched (do not use skip_layers on layer 0)")
        if holder._bound != (b, nqb, Hkv):
            raise RuntimeError(
                f"reuse_v1: cache bound to {holder._bound} but layer {layer_idx} "
                f"got (b,nqb,Hkv)=({b},{nqb},{Hkv}); prompt shape changed mid-sweep")

    label_L = holder.label[layer_idx].to(dev).bool()

    return reuse_v1_layer_per_hkv(
        query_states, key_states, value_states, label_L, holder.cache,
        holder.budget, holder.block_size, holder.segment_size, holder.causal,
        sink_blocks=holder.sink_blocks, local_blocks=holder.local_blocks,
        select_mode=holder.select_mode, top_p=holder.top_p,
        min_blocks=holder.min_blocks, max_blocks=holder.max_blocks,
    )


def get_reuse_v1_prefill(label_path, budget=32, block_size=128, segment_size=2048,
                         sink_blocks=1, local_blocks=2, causal=True, device='cuda',
                         select_mode='topk', top_p=0.9, min_blocks=8, max_blocks=64):
    """Build a per-model reuse_v1 prefill callable bound to a fresh holder.

    ``select_mode='topp'`` switches block selection from fixed top-k budget to
    nucleus (top-p) coverage; ``top_p`` / ``min_blocks`` / ``max_blocks`` tune it
    (min/max are INCLUSIVE of sink+diagonal+local; max_blocks None -> kernel
    headroom ``max_sel``).
    """
    # Ensure the repo root (with the pbs_attn package) is importable, then lazily
    # import the validated Triton kernel entry points.
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from pbs_attn.baselines.Reuse_v1 import reuse_v1_layer_per_hkv, IndexCache, _resolve_max_sel  # noqa: F401

    if budget not in (16, 32):
        raise ValueError(f"budget must be 16 or 32, got {budget}")
    if select_mode not in ('topk', 'topp'):
        raise ValueError(f"select_mode must be 'topk' or 'topp', got {select_mode!r}")

    holder = ReuseV1Holder(
        budget=budget, block_size=block_size, segment_size=segment_size,
        sink_blocks=sink_blocks, local_blocks=local_blocks, causal=causal,
        select_mode=select_mode, top_p=top_p, min_blocks=min_blocks, max_blocks=max_blocks,
    )
    holder.label = ReuseV1Holder.load_label(label_path, device)
    holder.num_layers = holder.label.shape[0]

    return functools.partial(reuse_v1_prefill, holder=holder)
