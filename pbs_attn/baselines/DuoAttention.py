"""DuoAttention as a prefill-stage sparse kernel for the pbs-attn_h patch framework.

Retrieval (full) kv-heads run dense causal ``flash_attn``; streaming kv-heads run
a sink+local block-sparse mask via ``block_streaming_attn_func`` (the same kernel
DuoAttention uses natively). Heads are split at the kv-head granularity (labels
are per kv-head), each group is attended independently, then outputs are
scattered back to the original head order.

Unlike DuoAttention's native eval (which compresses the KV cache during chunked
prefill / decode), this runs inside pbs-attn_h's single-pass prefill: streaming
heads simply get a sink+local mask over the full KV. Decode (q_len==1) is handled
by the standard dense forward, so this module only sees prefill (q_len>1) calls.
"""

import functools
import math

import torch

from pbs_attn.baselines.duo_pattern import load_duo_label

def _q_heads_for_kv(kv_idx, G):
    """Expand kv-head indices (K,) into their q-head indices (K*G,), GQA layout.

    q-head ``h`` attends kv-head ``h // G``, so kv-head ``j`` owns q-heads
    ``[j*G, j*G+1, ..., j*G+G-1]``.
    """
    return (kv_idx.unsqueeze(1) * G + torch.arange(G, device=kv_idx.device)).reshape(-1)


class DuoHolder:
    """Per-model static config + binarized label for the duo prefill path."""

    def __init__(self, label, sink_block_num, local_block_num,
                 block_size=128, causal=True, softmax_scale=None):
        self.label = label                    # (num_layers, Hkv) bool, True=full
        self.num_layers = int(label.shape[0])
        self.sink_block_num = int(sink_block_num)
        self.local_block_num = int(local_block_num)
        self.block_size = int(block_size)
        self.causal = bool(causal)
        self.softmax_scale = softmax_scale


def build_duo_holder(attn_load_dir, sink_size=None, recent_size=None,
                     threshold=0.5, sparsity=None, block_size=128,
                     causal=True, softmax_scale=None, device="cuda"):
    """Load the duo label + config and build a per-model holder.

    ``sink_size`` / ``recent_size`` default to the values in the label dir's
    config.json (128 / 256 for Llama-3.1-8B). They are converted to block counts
    for the block-sparse streaming kernel (block_size fixed at 128 there).
    """
    label, cfg_sink, cfg_recent, realized = load_duo_label(
        attn_load_dir, threshold=threshold, sparsity=sparsity, device=device)
    sink = int(sink_size) if sink_size is not None else cfg_sink
    recent = int(recent_size) if recent_size is not None else cfg_recent
    sink_block_num = max(1, math.ceil(sink / block_size))
    local_block_num = max(1, math.ceil(recent / block_size))
    n_full = int(label.sum().item())
    n_total = int(label.numel())
    print(f"🔧 DuoAttention: {n_full}/{n_total} full heads "
          f"(streaming sparsity={realized:.3f}), sink={sink}({sink_block_num}blk) "
          f"recent={recent}({local_block_num}blk)")
    return DuoHolder(label, sink_block_num, local_block_num,
                     block_size=block_size, causal=causal, softmax_scale=softmax_scale)


# __KERNEL_BODY__
def _streaming_blocksparse(q, k, v, sink_block_num, local_block_num, softmax_scale):
    """Sink+local block-sparse attention over (b, s, qheads, d) inputs (GQA ok).

    Packs to varlen layout and calls the block-sparse streaming kernel with
    ``head_mask_type = -1`` (streaming) per q-head and per-head streaming_info
    ``[sink_block_num, local_block_num]``.
    """
    from block_sparse_attn import block_streaming_attn_func

    b, s, qh, d = q.shape
    kvh = k.shape[2]
    q_u = q.reshape(b * s, qh, d)
    k_u = k.reshape(b * s, kvh, d)
    v_u = v.reshape(b * s, kvh, d)
    cu = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=q.device)
    head_mask_type = torch.full((qh,), -1, dtype=torch.int32, device=q.device)
    streaming_info = torch.tensor(
        [sink_block_num, local_block_num] * qh, dtype=torch.int32, device=q.device)
    o = block_streaming_attn_func(
        q_u, k_u, v_u, cu, cu, head_mask_type, streaming_info, s, s,
        p_dropout=0.0, softmax_scale=softmax_scale, is_causal=True)
    return o.reshape(b, s, qh, d)


def duo_attention_prefill(query_states, key_states, value_states,
                          num_key_value_groups=None, layer_idx=None, *, holder):
    """One-layer prefill entry for the HF patch.

    ``num_key_value_groups`` / ``layer_idx`` MUST stay explicit named params so
    ``patched_attention_forward`` detects them (via ``__code__.co_varnames``),
    passes ``layer_idx``, and keeps K/V native GQA (not repeat_kv'd).
    ``num_key_value_groups`` is only sanity-checked, not forwarded.

    Inputs are (b, H, s, d) / (b, Hkv, s, d); returns (b, H, s, d).
    """
    from flash_attn import flash_attn_func

    if layer_idx is None:
        raise RuntimeError("duo_attention_prefill requires layer_idx (patch must pass it)")

    b, H, s, d = query_states.shape
    Hkv = key_states.shape[1]
    G = H // Hkv
    if num_key_value_groups is not None and num_key_value_groups != G:
        raise ValueError(
            f"num_key_value_groups={num_key_value_groups} != H//Hkv={G}; "
            "K/V must be native GQA (not repeat_kv'd) for duo")
    if layer_idx >= holder.num_layers:
        raise IndexError(
            f"duo: layer_idx={layer_idx} out of range for label with "
            f"{holder.num_layers} layers")

    dev = query_states.device
    sm = holder.softmax_scale
    label_L = holder.label[layer_idx].to(dev)          # (Hkv,) bool, True=full
    out = query_states.new_empty((b, H, s, d))

    full_kv = torch.nonzero(label_L, as_tuple=False).flatten()
    stream_kv = torch.nonzero(~label_L, as_tuple=False).flatten()

    # FULL (retrieval) heads: dense causal flash attention.
    if full_kv.numel() > 0:
        qidx = _q_heads_for_kv(full_kv, G)
        q_f = query_states.index_select(1, qidx)
        k_f = key_states.index_select(1, full_kv)
        v_f = value_states.index_select(1, full_kv)
        o_f = flash_attn_func(
            q_f.transpose(1, 2), k_f.transpose(1, 2), v_f.transpose(1, 2),
            causal=holder.causal, softmax_scale=sm)     # (b, s, nf*G, d)
        out.index_copy_(1, qidx, o_f.transpose(1, 2))

    # STREAMING heads: sink+local block-sparse.
    if stream_kv.numel() > 0:
        qidx = _q_heads_for_kv(stream_kv, G)
        q_s = query_states.index_select(1, qidx)
        k_s = key_states.index_select(1, stream_kv)
        v_s = value_states.index_select(1, stream_kv)
        o_s = _streaming_blocksparse(
            q_s.transpose(1, 2).contiguous(),
            k_s.transpose(1, 2).contiguous(),
            v_s.transpose(1, 2).contiguous(),
            holder.sink_block_num, holder.local_block_num, sm)   # (b, s, ns*G, d)
        out.index_copy_(1, qidx, o_s.transpose(1, 2))

    return out


def get_duo_attention_prefill(attn_load_dir, sink_size=None, recent_size=None,
                              threshold=0.5, sparsity=None, block_size=128,
                              causal=True, softmax_scale=None, device="cuda"):
    """Build a per-model duo prefill callable bound to a fresh holder.

    ``sparsity`` (0..1) overrides ``threshold`` and picks a quantile so a
    ``sparsity`` fraction of heads become streaming (mirrors DuoAttention).
    """
    holder = build_duo_holder(
        attn_load_dir, sink_size=sink_size, recent_size=recent_size,
        threshold=threshold, sparsity=sparsity, block_size=block_size,
        causal=causal, softmax_scale=softmax_scale, device=device)
    return functools.partial(duo_attention_prefill, holder=holder)
