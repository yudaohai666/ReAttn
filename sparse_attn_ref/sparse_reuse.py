# SPDX-License-Identifier: MIT
"""Head-level cross-layer sparse-index reuse — core driver + HF integration.

Sections:
  1. Core driver  (SeqLayout, ReuseConfig, IndexCache, CrossLayerSparseReuse)
  2. HF Reuse     (ReuseHolder, load_model_with_reuse, sparse_reuse_attention_forward)

Hard constraints inherited from fmha_sm100:
  * topk in {16, 32}
  * page_size == 128 (tile index == page index only at this size)
  * anchor max_score numel must stay <= 2**31 (auto-split by the driver)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select

TOPK = 16
PAGE_SIZE = 128
_MAXSCORE_NUMEL_LIMIT = 1 << 31
_SPARSE_PREFILL_QLEN_THRESHOLD = 32


def _max_k_tiles(max_kv_len: int) -> int:
    return math.ceil(math.ceil(max_kv_len / 128) / 128) * 128


# ---------------------------------------------------------------------------
# 1. Core driver
# ---------------------------------------------------------------------------

@dataclass
class SeqLayout:
    """Token layout shared by every layer (T identical across layers)."""
    mode: str
    kv_lens: list[int]
    qo_offsets: list[int]
    page_size: int = PAGE_SIZE

    def __post_init__(self):
        assert self.mode in ("prefill", "decode")
        assert self.page_size == PAGE_SIZE, "page_size LOCKED to 128"
        if self.mode == "prefill":
            assert len(self.kv_lens) == 1, "prefill = single sequence"
        assert len(self.kv_lens) == len(self.qo_offsets)

    @property
    def T(self) -> int:
        if self.mode == "prefill":
            return self.kv_lens[0] - self.qo_offsets[0]
        return len(self.kv_lens)

    @property
    def max_kv_len(self) -> int:
        return max(self.kv_lens)

    @property
    def pages_per_row(self) -> list[int]:
        return [(L + self.page_size - 1) // self.page_size for L in self.kv_lens]

    @property
    def total_pages(self) -> int:
        return sum(self.pages_per_row)

    @property
    def num_valid_pages(self) -> int:
        return max(self.pages_per_row)


@dataclass
class ReuseConfig:
    num_layers: int
    num_kv_heads: int
    h_r: int
    head_dim: int = 128
    topk: int = TOPK
    page_size: int = PAGE_SIZE
    sink_blocks: int = 0
    local_blocks: int = 0
    reduce: str = "amax"
    check_input_valid: bool = False
    sm_scale: Optional[float] = None

    def __post_init__(self):
        assert self.topk in (16, 32), "kernel supports topk in {16, 32}"
        assert self.page_size == PAGE_SIZE, "page_size LOCKED to 128"
        assert self.reduce in ("amax", "mean")
        if self.sm_scale is None:
            self.sm_scale = 1.0 / math.sqrt(self.head_dim)

    @property
    def num_qo_heads(self) -> int:
        return self.num_kv_heads * self.h_r


class IndexCache:
    """Cross-layer persistent block indices (T, num_kv_heads, topk) int32."""

    def __init__(self, T: int, num_kv_heads: int, topk: int, device):
        self.buf = torch.full((T, num_kv_heads, topk), -1,
                              dtype=torch.int32, device=device)

    def refresh(self, kv_heads: torch.Tensor, idx: torch.Tensor):
        self.buf[:, kv_heads, :] = idx

    def reuse(self, kv_heads: torch.Tensor) -> torch.Tensor:
        return self.buf[:, kv_heads, :].contiguous()


def _qo_heads_for_kv(kv_heads: torch.Tensor, h_r: int) -> torch.Tensor:
    base = kv_heads.to(torch.int64).view(-1, 1) * h_r
    span = torch.arange(h_r, device=kv_heads.device).view(1, -1)
    return (base + span).reshape(-1)


def _layer_perm(is_anchor_row: torch.Tensor, h_r: int):
    """Per-layer head permutation [anchor heads | sparse heads].

    Returns (perm_kv, perm_q, n_anchor) on the same device as is_anchor_row.
    perm_q is the aligned 32-q-head order: perm_kv[j] -> q heads [j*h_r .. j*h_r+h_r).
    """
    anchor_h = is_anchor_row.nonzero(as_tuple=True)[0]
    sparse_h = (~is_anchor_row).nonzero(as_tuple=True)[0]
    perm_kv = torch.cat([anchor_h, sparse_h])
    perm_q = _qo_heads_for_kv(perm_kv, h_r)
    return perm_kv, perm_q, int(anchor_h.numel())


@torch.no_grad()
def _permute_linear_rows(linear, perm_heads: torch.Tensor, head_dim: int):
    """Reorder the OUTPUT rows of a Linear by head blocks (q/k/v_proj).

    weight: (n_heads*head_dim, hidden). View as (n_heads, head_dim, hidden),
    index by perm, flatten back. Bias (if any) reordered the same way.
    """
    W = linear.weight.data
    out_features, hidden = W.shape
    n_heads = out_features // head_dim
    Wv = W.view(n_heads, head_dim, hidden)[perm_heads]
    linear.weight.data = Wv.reshape(out_features, hidden).contiguous()
    if getattr(linear, "bias", None) is not None:
        b = linear.bias.data.view(n_heads, head_dim)[perm_heads]
        linear.bias.data = b.reshape(out_features).contiguous()


@torch.no_grad()
def _permute_linear_cols(linear, perm_q_heads: torch.Tensor, head_dim: int):
    """Reorder the INPUT columns of o_proj by q-head blocks.

    weight: (hidden, n_qheads*head_dim). View cols as (n_qheads, head_dim),
    index by perm_q (SAME direction as q_proj so the perms cancel). No bias
    reorder needed (o_proj bias is on the output/hidden dim).
    """
    W = linear.weight.data
    hidden, in_features = W.shape
    n_qheads = in_features // head_dim
    Wv = W.view(hidden, n_qheads, head_dim)[:, perm_q_heads]
    linear.weight.data = Wv.reshape(hidden, in_features).contiguous()


@torch.no_grad()
def reorder_model_weights(model, label: torch.Tensor, h_r: int, head_dim: int):
    """Fold per-layer head perm=[anchor|sparse] into q/k/v/o_proj weights.

    After this, projections emit q/k/v already in perm-space: the first
    n_anchor kv heads are anchors, the rest sparse. Returns a list of
    n_anchor per layer (in perm-space these are contiguous prefixes).

    Validated equivalent (verify_weight_reorder.py): o_proj reordered by the
    SAME perm_q as q_proj cancels exactly (negative control with inverse perm
    diverges by ~4.0).
    """
    base = model.model if hasattr(model, "model") else model
    layers = base.layers
    n_anchor_per_layer = []
    for L, lyr in enumerate(layers):
        perm_kv, perm_q, n_anchor = _layer_perm(label[L], h_r)
        attn = lyr.self_attn
        _permute_linear_rows(attn.q_proj, perm_q, head_dim)
        _permute_linear_rows(attn.k_proj, perm_kv, head_dim)
        _permute_linear_rows(attn.v_proj, perm_kv, head_dim)
        _permute_linear_cols(attn.o_proj, perm_q, head_dim)
        n_anchor_per_layer.append(n_anchor)
    return n_anchor_per_layer


def gather_heads(q, k_pages, v_pages, kv_heads, h_r):
    qo_ids = _qo_heads_for_kv(kv_heads, h_r)
    q_g = q.index_select(1, qo_ids).contiguous()
    k_g = k_pages.index_select(1, kv_heads).contiguous()
    v_g = v_pages.index_select(1, kv_heads).contiguous()
    return q_g, k_g, v_g, qo_ids


def slice_heads(q, k_pages, v_pages, h_lo, h_hi, h_r):
    """Zero-copy contiguous head slice for perm-space KV.

    Returns strided VIEWS of q[:, h_lo*h_r:h_hi*h_r], k/v_pages[:, h_lo:h_hi].
    fmha_sm100 accepts the dim=1 strided views directly (verified bit-identical,
    0 indexSelect kernels in verify_strided_kv.py). q is made contiguous because
    its head slice feeds the plan's packed layout; k/v stay as views.
    """
    qo_ids = torch.arange(h_lo * h_r, h_hi * h_r, device=q.device)
    q_g = q[:, h_lo * h_r:h_hi * h_r].contiguous()
    k_g = k_pages[:, h_lo:h_hi]
    v_g = v_pages[:, h_lo:h_hi]
    return q_g, k_g, v_g, qo_ids


class CrossLayerSparseReuse:
    """Stateful per-step driver. One instance per (layout, label table)."""

    def __init__(self, cfg: ReuseConfig, layout: SeqLayout,
                 label: torch.Tensor, device, reordered: bool = False):
        assert label.shape == (cfg.num_layers, cfg.num_kv_heads), label.shape
        assert bool(label[0].all()), "layer 0 must be all-anchor"
        self.cfg = cfg
        self.layout = layout
        self.device = device
        self.label = label.to(device)
        self.T = layout.T
        # reordered=True: weights were folded with perm=[anchor|sparse], so KV
        # arrives perm-space. anchor heads are the contiguous prefix [0:n_anchor]
        # and sparse heads the suffix; gather becomes a zero-copy slice.
        self.reordered = reordered
        self.index_cache = IndexCache(self.T, cfg.num_kv_heads, cfg.topk, device)

        self._anchor_h = []
        self._sparse_h = []
        self._n_anchor = []
        for L in range(cfg.num_layers):
            is_anchor = self.label[L]
            n_anchor = int(is_anchor.sum().item())
            self._n_anchor.append(n_anchor)
            if reordered:
                # perm-space: anchors are [0:n_anchor], sparse [n_anchor:].
                self._anchor_h.append(
                    torch.arange(n_anchor, device=device))
                self._sparse_h.append(
                    torch.arange(n_anchor, cfg.num_kv_heads, device=device))
            else:
                self._anchor_h.append(is_anchor.nonzero(as_tuple=True)[0])
                self._sparse_h.append((~is_anchor).nonzero(as_tuple=True)[0])

        self._dense_plan_cache: dict[int, object] = {}
        self._sparse_plan_cache: dict[int, object] = {}
        self._kv_indices = torch.arange(
            layout.total_pages, device=device, dtype=torch.int32)

    def _dense_plan(self, n_kv: int):
        Hq = n_kv * self.cfg.h_r
        if Hq in self._dense_plan_cache:
            return self._dense_plan_cache[Hq]
        lay = self.layout
        if lay.mode == "prefill":
            qo = torch.tensor([self.T], dtype=torch.int32)
            kv = torch.tensor([lay.kv_lens[0]], dtype=torch.int32)
            qoff = torch.tensor([lay.qo_offsets[0]], dtype=torch.int32)
        else:
            qo = torch.ones(self.T, dtype=torch.int32)
            kv = torch.tensor(lay.kv_lens, dtype=torch.int32)
            qoff = torch.tensor(lay.qo_offsets, dtype=torch.int32)
        plan = fmha_sm100_plan(
            qo, kv, Hq, num_kv_heads=n_kv, causal=True, qo_offset=qoff,
            page_size=self.cfg.page_size, output_maxscore=True)
        self._dense_plan_cache[Hq] = plan
        return plan

    def _sparse_plan(self, n_kv: int):
        if n_kv in self._sparse_plan_cache:
            return self._sparse_plan_cache[n_kv]
        lay = self.layout
        if lay.mode == "prefill":
            qo = torch.tensor([self.T], dtype=torch.int32)
            kv = torch.tensor([lay.kv_lens[0]], dtype=torch.int32)
            qoff = torch.tensor([lay.qo_offsets[0]], dtype=torch.int32)
        else:
            qo = torch.ones(self.T, dtype=torch.int32)
            kv = torch.tensor(list(lay.kv_lens), dtype=torch.int32)
            qoff = torch.tensor(list(lay.qo_offsets), dtype=torch.int32)
        plan = fmha_sm100_plan(
            qo, kv, n_kv * self.cfg.h_r, num_kv_heads=n_kv, causal=True,
            qo_offset=qoff, page_size=self.cfg.page_size, kv_block_num=self.cfg.topk)
        self._sparse_plan_cache[n_kv] = plan
        return plan

    def anchor_dense_with_index(self, q_a, k_a, v_a, n_anchor):
        h_r = self.cfg.h_r
        max_kt = _max_k_tiles(self.layout.max_kv_len)
        full_numel = (n_anchor * h_r) * max_kt * self.T

        # Group BOUNDARIES as plain Python ints (CPU). Avoid grp[0].item()
        # GPU->CPU syncs in the hot loop: the splits of arange(n_anchor) have
        # deterministic integer starts we can track without touching the GPU.
        if full_numel <= _MAXSCORE_NUMEL_LIMIT:
            group_bounds = [(0, n_anchor)]              # (g0, ng)
        else:
            per_head = h_r * max_kt * self.T
            g = max(1, _MAXSCORE_NUMEL_LIMIT // per_head)
            group_bounds = []
            s = 0
            while s < n_anchor:
                ng = min(g, n_anchor - s)
                group_bounds.append((s, ng))
                s += ng

        o_a = torch.empty(self.T, n_anchor * h_r, self.cfg.head_dim,
                          device=self.device, dtype=torch.bfloat16)
        idx_a = torch.empty(self.T, n_anchor, self.cfg.topk,
                            dtype=torch.int32, device=self.device)
        for g0, ng in group_bounds:
            if self.reordered:
                # anchor heads already contiguous; pure-int slices, no sync.
                qg = q_a[:, g0 * h_r:(g0 + ng) * h_r].contiguous()
                kg = k_a[:, g0:g0 + ng]
                vg = v_a[:, g0:g0 + ng]
            else:
                grp = torch.arange(g0, g0 + ng, device=self.device)
                qo_ids = _qo_heads_for_kv(grp, h_r)
                qg = q_a.index_select(1, qo_ids).contiguous()
                kg = k_a.index_select(1, grp).contiguous()
                vg = v_a.index_select(1, grp).contiguous()
            plan = self._dense_plan(ng)
            o_g, ms = fmha_sm100(
                qg, kg, vg, plan, sm_scale=self.cfg.sm_scale,
                kv_indices=self._kv_indices, output_o=True, output_maxscore=True)
            assert ms is not None, "max_score=None: 2**31 silent degrade"
            _, K, Tq = ms.shape
            ms_kv = ms.view(ng, h_r, K, Tq)
            ms_kv = ms_kv.amax(1) if self.cfg.reduce == "amax" else ms_kv.mean(1)
            ms_kv = ms_kv.contiguous()
            idx_g = sparse_topk_select(
                ms_kv, self.cfg.topk, num_valid_pages=self.layout.num_valid_pages,
                force_begin_blocks=self.cfg.sink_blocks,
                force_end_blocks=self.cfg.local_blocks)
            if self.reordered:
                o_a[:, g0 * h_r:(g0 + ng) * h_r] = o_g
                idx_a[:, g0:g0 + ng] = idx_g
            else:
                grp = torch.arange(g0, g0 + ng, device=self.device)
                qo_ids = _qo_heads_for_kv(grp, h_r)
                o_a.index_copy_(1, qo_ids, o_g)
                idx_a.index_copy_(1, grp, idx_g)
        return o_a, idx_a

    def run_layer(self, L: int, q, k_pages, v_pages) -> torch.Tensor:
        cfg = self.cfg
        anchor_h = self._anchor_h[L]
        sparse_h = self._sparse_h[L]
        n_anchor = self._n_anchor[L]
        out = torch.empty(self.T, cfg.num_qo_heads, cfg.head_dim,
                          device=self.device, dtype=torch.bfloat16)

        if anchor_h.numel():
            if self.reordered:
                # anchors = contiguous prefix [0:n_anchor]; zero-copy slices.
                q_a, k_a, v_a, qo_ids_a = slice_heads(
                    q, k_pages, v_pages, 0, n_anchor, cfg.h_r)
            else:
                q_a, k_a, v_a, qo_ids_a = gather_heads(
                    q, k_pages, v_pages, anchor_h, cfg.h_r)
            o_a, idx_a = self.anchor_dense_with_index(q_a, k_a, v_a, int(anchor_h.numel()))
            if self.reordered:
                out[:, :n_anchor * cfg.h_r] = o_a
                self.index_cache.buf[:, :n_anchor, :] = idx_a
            else:
                out.index_copy_(1, qo_ids_a, o_a)
                self.index_cache.refresh(anchor_h, idx_a)

        if sparse_h.numel():
            if self.reordered:
                # sparse = contiguous suffix [n_anchor:]; zero-copy slices.
                q_s, k_s, v_s, qo_ids_s = slice_heads(
                    q, k_pages, v_pages, n_anchor, cfg.num_kv_heads, cfg.h_r)
                idx_s = self.index_cache.buf[:, n_anchor:, :].contiguous()
            else:
                q_s, k_s, v_s, qo_ids_s = gather_heads(
                    q, k_pages, v_pages, sparse_h, cfg.h_r)
                idx_s = self.index_cache.reuse(sparse_h)
            plan = self._sparse_plan(int(sparse_h.numel()))
            o_s, _ = fmha_sm100(
                q_s, k_s, v_s, plan, sm_scale=cfg.sm_scale,
                kv_indices=self._kv_indices,
                kv_block_indexes=idx_s, check_input_valid=cfg.check_input_valid)
            if self.reordered:
                out[:, n_anchor * cfg.h_r:] = o_s
            else:
                out.index_copy_(1, qo_ids_s, o_s)
        return out


# ---------------------------------------------------------------------------
# 2. HF Reuse (online sparse inference)
# ---------------------------------------------------------------------------

def _dense_fallback(query, k_pages, v_pages, S_kv, scaling):
    """Full dense attention over paged KV, for the sparse-inactive phase.

    KV already arrives in paged layout (num_pages, H_kv, page_size, D) from the
    PagedSparseReuseCache, so there is no _to_paged conversion: we feed the
    paged buffers straight to fmha_sm100. No topk / max_score; just dense O.
    """
    q_sd = query[0].transpose(0, 1).contiguous()   # (S_q, Hq, D)
    S_q = q_sd.shape[0]
    Hq = q_sd.shape[1]
    Hkv = k_pages.shape[1]
    total_pages = k_pages.shape[0]
    is_prefill = S_q > 1

    if is_prefill:
        qo = torch.tensor([S_q], dtype=torch.int32)
        kv = torch.tensor([S_kv], dtype=torch.int32)
        qoff = torch.tensor([0], dtype=torch.int32)
    else:
        qo = torch.ones(1, dtype=torch.int32)
        kv = torch.tensor([S_kv], dtype=torch.int32)
        qoff = torch.tensor([S_kv - 1], dtype=torch.int32)

    plan = fmha_sm100_plan(qo, kv, Hq, num_kv_heads=Hkv, causal=True, qo_offset=qoff,
                           page_size=PAGE_SIZE, output_maxscore=False)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=query.device)
    o, _ = fmha_sm100(q_sd, k_pages, v_pages, plan, sm_scale=float(scaling),
                      kv_indices=kv_indices, output_o=True, output_maxscore=False)
    return o.unsqueeze(0), None


@dataclass
class ReuseHolder:
    label: torch.Tensor
    cfg: ReuseConfig
    sparse_phase: str = "both"
    driver: Optional[CrossLayerSparseReuse] = None
    cache: Optional[object] = None  # PagedSparseReuseCache
    reordered: bool = False  # weights folded with perm=[anchor|sparse]


_HOLDER: Optional[ReuseHolder] = None


def _phase_active(sparse_phase: str, is_prefill: bool) -> bool:
    if sparse_phase == "both":
        return True
    if sparse_phase == "prefill":
        return is_prefill
    return not is_prefill


def sparse_reuse_attention_forward(module, query, key, value, attention_mask,
                                   scaling, dropout=0.0, **kwargs):
    h = _HOLDER
    assert h is not None, "ReuseHolder not initialised -- call load_model_with_reuse"
    assert query.shape[0] == 1, "sparse_reuse path is B==1 only"

    # KV always arrives in paged layout (num_pages, H_kv, page_size, D) from the
    # PagedSparseReuseCache; effective length comes from the cache.
    L = int(module.layer_idx)
    S_kv = int(h.cache.layers[L].get_seq_length())
    S_q = query.shape[-2]
    is_prefill = S_q > 1

    if not _phase_active(h.sparse_phase, is_prefill):
        return _dense_fallback(query, key, value, S_kv, scaling)

    if L == 0:
        if is_prefill:
            layout = SeqLayout(mode="prefill", kv_lens=[S_kv], qo_offsets=[0])
        else:
            layout = SeqLayout(mode="decode", kv_lens=[S_kv], qo_offsets=[S_kv - 1])
        h.cfg.sm_scale = float(scaling)
        h.driver = CrossLayerSparseReuse(h.cfg, layout, h.label, query.device,
                                         reordered=h.reordered)

    drv = h.driver
    assert drv is not None and L < h.cfg.num_layers

    q_sd = query[0].transpose(0, 1).contiguous()
    out = drv.run_layer(L, q_sd, key, value)
    return out.unsqueeze(0), None


def load_model_with_reuse(model_path: str, label_path: str,
                          sparse_phase: str = "both", reduce: str = "amax",
                          sink_blocks: int = 0, local_blocks: int = 0,
                          topk: int = TOPK,
                          dtype=torch.bfloat16, device="cuda",
                          paged_cache_max_kv_len: int = 128,
                          reorder_weights: bool = False):
    global _HOLDER
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("sparse_reuse", sparse_reuse_attention_forward)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="sparse_reuse", torch_dtype=dtype)
    model = model.to(device).eval()

    c = model.config
    num_layers = c.num_hidden_layers
    num_kv = c.num_key_value_heads
    h_r = c.num_attention_heads // num_kv
    head_dim = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)

    label_f = torch.load(label_path, map_location="cpu")
    assert tuple(label_f.shape) == (num_layers, num_kv), \
        f"label {tuple(label_f.shape)} != ({num_layers},{num_kv})"
    label = (label_f > 0.5).to(device)
    assert bool(label[0].all()), "layer 0 must be all-anchor"

    if reorder_weights:
        # Fold per-layer perm=[anchor|sparse] into q/k/v/o_proj so projections
        # emit perm-space KV; driver then uses zero-copy contiguous slices
        # instead of gather_heads' index_select. Validated equivalent
        # (verify_weight_reorder.py). The label stays in ORIGINAL space but the
        # driver, when reordered=True, treats anchors as the prefix [0:n_anchor]
        # (n_anchor is perm-invariant per layer).
        reorder_model_weights(model, label, h_r, head_dim)
        print(f"[sparse_reuse] weights reordered to perm-space [anchor|sparse] "
              f"per layer (index_select -> zero-copy slice)")

    cfg = ReuseConfig(num_layers=num_layers, num_kv_heads=num_kv, h_r=h_r,
                      head_dim=head_dim, topk=topk, page_size=PAGE_SIZE,
                      reduce=reduce, check_input_valid=False,
                      sink_blocks=sink_blocks, local_blocks=local_blocks)
    _HOLDER = ReuseHolder(label=label, cfg=cfg, sparse_phase=sparse_phase,
                          reordered=reorder_weights)
    n_anchor = int(label.sum().item())
    print(f"[sparse_reuse] layers={num_layers} kv_heads={num_kv} h_r={h_r} "
          f"head_dim={head_dim} | anchor slots {n_anchor}/{label.numel()} "
          f"({n_anchor/label.numel()*100:.1f}%) phase={sparse_phase} reduce={reduce}")

    from .paged_cache import PagedSparseReuseCache
    assert paged_cache_max_kv_len > 0, "paged_cache_max_kv_len must be > 0"
    cache = PagedSparseReuseCache(
        num_layers=num_layers,
        max_kv_len=paged_cache_max_kv_len,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        page_size=PAGE_SIZE,
        dtype=dtype,
        device=device,
    )
    _HOLDER.cache = cache
    print(f"[sparse_reuse] PagedSparseReuseCache: max_pages={cache.max_pages} "
          f"(max_kv_len={paged_cache_max_kv_len})")
    return model, tok, cache
