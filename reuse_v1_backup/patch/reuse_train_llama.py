"""DuoAttention-style trainable head-label learning for the reuse_v1 scheme.

Learns a per-(layer, kv-head) **anchor vs sparse** label with a soft gate, using
the same recipe as DuoAttention (two-way distillation + L1 sparsity), but the
"cheap" student branch is the reuse_v1 block-sparse attention instead of a
static sink+recent stream.

Per layer, per kv-head:
  * teacher / anchor branch (no_grad): dense block attention that also emits exact
    per-(q_block,k_block) scores -> this layer's top-k selection (stored for the
    NEXT layer to reuse). Uses ``block_sparse_attn_with_score``.
  * student cheap branch (grad): block-sparse attention that REUSES the previous
    layer's top-k selection for the same kv-head (training proxy for the inference
    "most-recent anchor" reuse). Uses ``sparse_block_attn_trainable_from_cache``.
  * blend: ``out = (1-g)*sparse + g*dense.detach()`` with ``g`` = per-kv-head gate.

Gate polarity matches DuoAttention: g=1 -> anchor (expensive), g=0 -> sparse
(cheap); L1 pushes g -> 0. Layer 0 is forced all-anchor (gate frozen at 1, no
sparse branch), matching the reuse_v1 inference constraint ``label[0].all()``.

Hyperparameters are fixed per run and MUST match inference:
    select_mode='topk', budget=32, block_size=128, sink_blocks=1, local_blocks=2.
"""

import os
import sys
import types

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
)

# NOTE: no tuple_kv_cache here. This is a prefill-only two-way forward (full
# sequence, no generation / no KV reuse), so the DuoAttention-style tuple KV
# cache is unnecessary. It is also incompatible with this transformers version
# (5.6.0): the model/layer/attention call conventions below are the native ones.

# Make the validated pure-Triton reuse kernels importable. They live in
# ``<repo>/pbs_attn/baselines`` and use bare top-level imports (``import
# Reuse_v1``), so that directory must be on sys.path.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_BASELINES = os.path.join(_REPO_ROOT, "pbs_attn", "baselines")
_RV1_DIR = os.path.join(_REPO_ROOT, "reuse_v1")
for _p in (_REPO_ROOT, _BASELINES, _RV1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Reuse_v1 import (
    block_sparse_attn_with_score,
    block_sparse_attn_dense_no_score,
    select_topk_blocks_per_kv_head,
)
from Reuse_v1_bwd import sparse_block_attn_trainable_from_cache

# Ulysses sequence-parallel all-to-all primitive (no-op when sp_size==1).
from sp_ulysses import SPContext, seq_to_head, head_to_seq

# HardKuma stochastic gate (opt-in --reg_mode kuma). The reparam z is recomputed
# INSIDE each layer forward from a FROZEN per-step noise so the (a,b) gradient
# stays consumed inside the FSDP-hooked layer forward (per-layer reduce-scatter
# AVERAGE), matching the distill ``* world_size`` recipe; and so activation-
# checkpoint recompute deterministically reproduces the same z.
from kuma_gate import hardkuma_sample_z, KUMA_SUPPORT
# DDP (deterministic differentiable pruning) gate: the training-forward blend
# gate is a straight-through clamp of the anchor_heads LOGIT (z_loga). The raw
# logit is left unbounded above (only floored per-step in the trainer); the STE
# lets the blend see a [0,1] value while the gradient flows to the raw logit.
from ddp_gate import ste_clamp
# Hard Concrete gate (opt-in --reg_mode hc). The reparam z is recomputed INSIDE
# each layer forward from a FROZEN per-step uniform noise (holder.hc_noise) and
# the trainable log_alpha stored in anchor_heads. Same FSDP/AC protocol as kuma.
from hc_gate import hc_sample_z
# HardLogistic(mu, s) gate (opt-in --reg_mode hln). HC with learnable global
# temperature s replacing the fixed beta. Noise is frozen Uniform (same as HC).
# density = sigmoid(mu), s-free; export threshold mu > 0.
from hln_gate import hl_sample_z, hl_s_from_raw, NOISE_EPS as HL_NOISE_EPS

class ReuseV1TrainHolder:
    """Per-model training state that threads the top-k selection across layers.

    ``sels[L] = (k_sel, k_cnt)`` is the fresh selection computed by layer L's
    dense/teacher stream (shape ``(b, Hkv, nqb, max_sel)`` / ``(b, Hkv, nqb)``).
    Layer L+1's sparse/student branch reuses ``sels[L]``. Reset at layer 0 (the
    reliable "new forward" signal), exactly like the inference holder.
    """

    def __init__(self, budget, block_size, segment_size, sink_blocks,
                 local_blocks, causal=True, select_mode="topk", top_p=0.9,
                 min_blocks=8, max_blocks=64):
        self.budget = budget
        self.block_size = block_size
        self.segment_size = segment_size
        self.sink_blocks = sink_blocks
        self.local_blocks = local_blocks
        self.causal = causal
        # Student sparse-branch block selection. "topk" (default) = fixed
        # ``budget`` blocks (byte-identical to the original path). "topp" =
        # nucleus: keep the fewest top-scored blocks whose cumulative mean-mass
        # >= ``top_p``, count clamped to [min_blocks, max_blocks]; ``budget`` is
        # then only the cache-fallback width. These MUST match the reuse_v1
        # inference selection or the exported head label will not transfer.
        self.select_mode = select_mode
        self.top_p = top_p
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.sels = {}
        # Forward mode set by the train loop; defaults to single-pass behavior:
        #   "single"       one-pass, grad in the same b=2 forward.
        #   "fill"         legacy b=2 [teacher|student] no_grad fill (kept for the
        #                  --two_pass path when memory allows folding).
        #   "fill_teacher" b=1 no_grad teacher-only fill (distill target); reads
        #                  nothing, writes nothing (Option A: halves fill memory).
        #   "fill_student" b=1 no_grad student-only fill; resets at L0 and records
        #                  every layer's top-k selection into holder.sels.
        #   "use"          b=1 grad student-only pass; reads frozen sels,
        #                  recompute-safe under activation checkpointing.
        self.phase = "single"
        # Regularization / gate mode. "l1" (default) = deterministic sigmoid-free
        # gate (anchor_heads used directly, clamped to [0,1]) + L1 sparsity. "kuma"
        # = HardKuma stochastic gate: per-layer trainable (kuma_a, kuma_b) shape
        # params, z reparameterized inside the layer forward from frozen noise, +
        # Lagrangian density constraint (driven in the train loop). Set by
        # enable_llama_reuse_v1_training.
        self.reg_mode = "l1"
        # kuma mode only: frozen per-step reparam NOISE, shape (num_layers, Hkv),
        # drawn ONCE at the top of each train step (outside every forward / AC
        # region) with a rank-shared seed. Read inside each layer's blend as
        # ``kuma_noise[L]`` so the SAME u reproduces z under AC recompute and
        # across the fill/use passes. None until the train loop populates it.
        self.kuma_noise = None
        # stg mode only: frozen per-step Gaussian noise, shape (num_layers, Hkv),
        # drawn ONCE at the top of each train step (same pattern as kuma_noise).
        # Read inside each layer's blend as ``stg_noise[L]``.
        self.stg_noise = None
        # hln mode only: frozen per-step Uniform noise, shape (num_layers, Hkv),
        # drawn ONCE at the top of each train step (same pattern as kuma_noise).
        # HardLogistic(mu, s) uses logistic noise (uniform -> logit), same as HC.
        self.hln_noise = None
        # hc / hcl mode only: frozen per-step Uniform noise, shape (num_layers, Hkv),
        # drawn ONCE per step outside all forward/AC regions (same pattern as kuma/hln).
        # hcl shares the same noise + sampling as hc; only the Lagrangian density differs.
        self.hc_noise = None
        # Ulysses sequence parallelism. sp.sp_size==1 -> disabled (the entire SP
        # code path in the attention forward is gated on sp.sp_size>1, so the
        # non-SP behavior is byte-for-byte unchanged). Set via
        # enable_sequence_parallel(model, group).
        self.sp = SPContext(None)

    def reset(self):
        self.sels = {}
        # kuma mode: per-kv-head anchor cache, mirrors inference IndexCache semantics.
        # anchor_cache[h] = (k_sel, k_cnt) from the most recent layer where
        # kv-head h was sampled as anchor (z > 0.5). Populated during fill_student,
        # read during use pass. Layer 0 (all-anchor) always writes all slots, so
        # every head has a valid entry from layer 0 onward.
        self.anchor_cache = {}


def _project_qkv(self, hidden, H, Hkv, hd, cos, sin):
    """Project + RoPE a single stream. Returns q/k/v in (b, H|Hkv, s, d)."""
    b, s, _ = hidden.size()
    q = self.q_proj(hidden).view(b, s, H, hd).transpose(1, 2)
    k = self.k_proj(hidden).view(b, s, Hkv, hd).transpose(1, 2)
    v = self.v_proj(hidden).view(b, s, Hkv, hd).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)  # (b,H,s,d) layout, unsqueeze_dim=1
    return q.contiguous(), k.contiguous(), v.contiguous()


def _teacher_dense(self, teacher_hidden, H, Hkv, hd, cos, sin, holder, hidden_size):
    """no_grad dense over ALL heads -> the distillation target o_proj output.

    Uses the score-FREE dense kernel: the teacher discards the per-block score,
    so skipping its emission avoids the ~12.6GB fp32 mass scratch (bit-identical
    attention output).
    """
    bsz, q_len, _ = teacher_hidden.size()
    with torch.no_grad():
        qt, kt, vt = _project_qkv(self, teacher_hidden, H, Hkv, hd, cos, sin)
        out_t_h = block_sparse_attn_dense_no_score(
            qt, kt, vt,
            block_size=holder.block_size, segment_size=holder.segment_size,
            causal=holder.causal,
        )
        out_t = out_t_h.transpose(1, 2).reshape(bsz, q_len, hidden_size)
        return self.o_proj(out_t)


def _student_dense_and_select(self, student_hidden, H, Hkv, hd, G, cos, sin, holder):
    """no_grad: student dense pass + top-k block selection.

    Returns ``(out_dense_s (b,s,H,d), (k_sel, k_cnt))``. ``out_dense_s`` is the
    anchor-branch value used in the blend; the selection feeds the NEXT layer's
    sparse branch.
    """
    bsz = student_hidden.size(0)
    with torch.no_grad():
        qs, ks, vs = _project_qkv(self, student_hidden, H, Hkv, hd, cos, sin)
        full_bm = torch.empty((bsz, H, 1, 1), dtype=torch.bool, device=qs.device)
        out_dense_h, block_score = block_sparse_attn_with_score(
            qs, ks, vs, full_bm,
            block_size=holder.block_size, segment_size=holder.segment_size,
            causal=holder.causal,
        )
        k_sel, k_cnt = select_topk_blocks_per_kv_head(
            block_score, G, holder.budget,
            sink_blocks=holder.sink_blocks, local_blocks=holder.local_blocks,
            causal=holder.causal, select_mode=holder.select_mode,
            top_p=holder.top_p, min_blocks=holder.min_blocks,
            max_blocks=holder.max_blocks,
        )
        return out_dense_h.transpose(1, 2), (k_sel, k_cnt)


def _student_dense_only(self, student_hidden, H, Hkv, hd, cos, sin, holder):
    """no_grad: student dense pass WITHOUT selection (blend anchor value only).

    Used by the ``use`` (grad) pass, which reads a FROZEN selection and thus
    does not need to recompute per-block scores. Uses the score-FREE dense
    kernel so the recomputed forward under activation checkpointing does NOT
    allocate the ~12.6GB fp32 mass scratch -- this is what lets 128k per-layer
    AC fit in backward with no CPU offload. Bit-identical to the score kernel's
    attention output. Returns ``(b,s,H,d)``.
    """
    with torch.no_grad():
        qs, ks, vs = _project_qkv(self, student_hidden, H, Hkv, hd, cos, sin)
        out_dense_h = block_sparse_attn_dense_no_score(
            qs, ks, vs,
            block_size=holder.block_size, segment_size=holder.segment_size,
            causal=holder.causal,
        )
        return out_dense_h.transpose(1, 2)


def _kv_head_gate(self, holder, lo=None, hi=None):
    """Per-kv-head blend gate values in [0,1] for the requested kv-head slice.

    * ``l1`` mode: deterministic ``anchor_heads.clamp(0,1)``.
    * ``kuma`` mode: HardKuma reparam ``z`` recomputed HERE from the frozen
      per-step noise ``holder.kuma_noise[L]`` and this layer's trainable
      ``(kuma_a, kuma_b)``. Recomputing inside the layer forward (rather than
      pre-sampling z outside) keeps the (a,b) gradient consumed inside the
      FSDP-hooked forward -> per-layer reduce-scatter AVERAGE, which the distill
      ``* world_size`` recipe compensates unchanged; and the frozen noise makes
      z deterministic under activation-checkpoint recompute + the fill/use passes.

    Layer 0 never reaches a blend site (it is all-anchor), so no special-casing
    is needed here. ``lo/hi`` slice this rank's kv-head range under Ulysses SP.
    """
    if holder.reg_mode == "kuma":
        L = self._reuse_layer_idx
        a = self.kuma_a.clamp(1e-6, 1e6)
        b = self.kuma_b.clamp(1e-6, 1e6)
        u = holder.kuma_noise[L].to(a.device)
        if lo is not None:
            a, b, u = a[lo:hi], b[lo:hi], u[lo:hi]
        return hardkuma_sample_z(u, a, b, KUMA_SUPPORT)
    if holder.reg_mode == "ddp":
        # DDP: anchor_heads IS the logit z_loga; the blend gate is a
        # straight-through clamp to [0,1] (forward value clamped, gradient
        # passes to the raw logit, which is left free to exceed 1). Consumed
        # inside the FSDP-hooked forward exactly like the l1/kuma gates, so the
        # per-layer reduce-scatter AVERAGE + distill ``* world_size`` recipe is
        # unchanged. Layer 0 never blends (all-anchor), so no special-casing.
        z = self.anchor_heads
        if lo is not None:
            z = z[lo:hi]
        return ste_clamp(z, 0.0, 1.0)
    if holder.reg_mode == "stg":
        # STG: anchor_heads is mu_code; blend gate = clip(mu + 0.5 + sigma*eps).
        # eps is frozen per-step noise from holder.stg_noise[L], same pattern as
        # kuma. Recomputing inside the layer forward keeps mu gradient in the
        # FSDP-hooked forward (reduce-scatter AVERAGE, compensated by * world_size).
        L = self._reuse_layer_idx
        mu = self.anchor_heads
        eps = holder.stg_noise[L].to(mu.device)
        sigma = getattr(holder, "stg_sigma", 0.5)
        if lo is not None:
            mu, eps = mu[lo:hi], eps[lo:hi]
        return torch.clamp(mu + 0.5 + sigma * eps, 0.0, 1.0)
    if holder.reg_mode in ("hc", "hcl"):
        # Hard Concrete: anchor_heads is log_alpha; blend gate = hc_sample_z(u, log_alpha).
        # u is frozen per-step noise from holder.hc_noise[L], same protocol as kuma.
        # Recomputing inside the layer forward keeps log_alpha gradient in the
        # FSDP-hooked forward (reduce-scatter AVERAGE, compensated by * world_size).
        # hcl uses the same HC gate sampling; only the Lagrangian density formula
        # differs (sigmoid(alpha) = P(z>0.5) instead of sigmoid(alpha+1.599) = P(z>0)).
        L = self._reuse_layer_idx
        log_alpha = self.anchor_heads
        u = holder.hc_noise[L].to(log_alpha.device)
        if lo is not None:
            log_alpha, u = log_alpha[lo:hi], u[lo:hi]
        return hc_sample_z(u, log_alpha)
    if holder.reg_mode == "hln":
        # HardLogistic(mu, s): anchor_heads is mu; s is a global learnable
        # temperature stored as holder.hl_s_raw (nn.Parameter, unconstrained).
        # Blend gate = hl_sample_z(u, mu, s) recomputed inside layer forward
        # from frozen per-step uniform noise holder.hln_noise[L], same protocol
        # as hc_noise. Gradient flows to mu (and s via holder.hl_s_raw).
        L = self._reuse_layer_idx
        mu = self.anchor_heads
        u = holder.hln_noise[L].to(mu.device)
        s_raw = getattr(holder, "hl_s_raw", None)
        s = hl_s_from_raw(s_raw.to(mu.device)) if s_raw is not None else 0.5
        if lo is not None:
            mu, u = mu[lo:hi], u[lo:hi]
        return hl_sample_z(u, mu, s)
    g = self.anchor_heads.clamp(0, 1)
    if lo is not None:
        g = g[lo:hi]
    return g


def _anchor_cache_sel(holder, Hkv, sel_fallback, L, lo=None, hi=None):
    """Return the per-kv-head anchor selection to use for layer L's sparse blend.

    kuma mode: assemble (k_sel, k_cnt) from holder.anchor_cache. Each slot
    holds the most-recent anchor write for that kv-head id, mirroring inference
    IndexCache semantics. Layer 0 always writes all slots on reset, so every
    head is guaranteed to have a valid entry.

    non-kuma mode: return sel_fallback unchanged (original sels[L-1] behaviour).

    SP mode (lo/hi provided): only assemble this rank's local kv-head slice
    [lo, hi) from anchor_cache. The SP blend helpers operate on local heads only,
    so the returned sel has shape (b, Hkv_local, nqb, max_sel).
    """
    if holder.reg_mode != "kuma" or not holder.anchor_cache:
        return sel_fallback
    h_start = lo if lo is not None else 0
    h_end   = hi if hi is not None else Hkv
    k_sel_list = [holder.anchor_cache[h][0] for h in range(h_start, h_end)]
    k_cnt_list = [holder.anchor_cache[h][1] for h in range(h_start, h_end)]
    return torch.cat(k_sel_list, dim=1), torch.cat(k_cnt_list, dim=1)


def _student_sparse_blend(self, student_hidden, out_dense_s, sel_prev,
                          H, Hkv, hd, G, cos, sin, holder, hidden_size):
    """Grad-capable sparse branch reusing ``sel_prev`` + per-kv-head gate blend.

    ``blended = (1-g)*sparse + g*dense``. Grad-ness is controlled by the caller
    (wrap in ``torch.no_grad()`` for the fill pass). Returns ``(b,q_len,hidden)``.
    """
    bsz, q_len, _ = student_hidden.size()
    qs, ks, vs = _project_qkv(self, student_hidden, H, Hkv, hd, cos, sin)
    k_sel_prev, k_cnt_prev = sel_prev
    # (b, Hkv, nqb, max_sel) -> (b, nqb, Hkv, max_sel) as the sparse kernel wants
    k_sel_in = k_sel_prev.permute(0, 2, 1, 3).contiguous()
    k_cnt_in = k_cnt_prev.permute(0, 2, 1).contiguous()
    out_sparse_h = sparse_block_attn_trainable_from_cache(
        qs, ks, vs, k_sel_in, k_cnt_in,
        budget=holder.budget, block_size=holder.block_size, causal=holder.causal,
    )
    out_sparse = out_sparse_h.transpose(1, 2)  # (b, s, H, d)
    # kuma gate is float32 (kuma_a/kuma_b are float32 Params); cast to the bf16
    # attention output so the blend stays bf16 for o_proj. No-op for l1 (bf16).
    g = _kv_head_gate(self, holder).to(out_sparse.dtype).repeat_interleave(G).view(1, 1, H, 1)
    blended = (1 - g) * out_sparse + g * out_dense_s  # out_dense_s detached
    return self.o_proj(blended.reshape(bsz, q_len, hidden_size))


# =====================================================================
# Ulysses sequence-parallel helpers.
#
# Layout convention inside SP attention: project q/k/v on the LOCAL sequence
# shard (all heads), all-to-all to FULL sequence / LOCAL head subset, run the
# reuse kernels + gate blend there (each rank sees the whole sequence for its
# head subset -> global top-k selection stays exact), then all-to-all back to
# LOCAL sequence / all heads and o_proj. All SP helpers return the attention
# output in (b, H_local, s_full, d) BEFORE the all-to-all-back + o_proj, which
# the single tail ``_sp_finish`` performs.
# =====================================================================

def _sp_project_a2a(self, hidden, H, Hkv, hd, cos, sin, group):
    """Project local-seq hidden (all heads) + RoPE, all-to-all to full-seq /
    local-head. Returns qf (b,H_local,s_full,d), kf/vf (b,Hkv_local,s_full,d)."""
    q, k, v = _project_qkv(self, hidden, H, Hkv, hd, cos, sin)  # (b,*,s_local,d)
    return seq_to_head(q, group), seq_to_head(k, group), seq_to_head(v, group)


def _sp_finish(self, attn_out_h, bsz, hidden_size, group):
    """(b,H_local,s_full,d) -> all-to-all-back (b,H,s_local,d) -> o_proj."""
    out = head_to_seq(attn_out_h, group)  # (b, H, s_local, d)
    s_local = out.shape[2]
    out = out.transpose(1, 2).reshape(bsz, s_local, hidden_size)
    return self.o_proj(out)


def _sp_teacher_dense(self, hidden, H, Hkv, hd, cos, sin, holder, hidden_size):
    """no_grad SP dense teacher -> distill target (b, s_local, hidden)."""
    bsz = hidden.size(0)
    group = holder.sp.group
    with torch.no_grad():
        qf, kf, vf = _sp_project_a2a(self, hidden, H, Hkv, hd, cos, sin, group)
        out_h = block_sparse_attn_dense_no_score(
            qf, kf, vf, block_size=holder.block_size,
            segment_size=holder.segment_size, causal=holder.causal)
        return _sp_finish(self, out_h, bsz, hidden_size, group)


def _sp_student_dense_and_select(self, hidden, H, Hkv, hd, G, cos, sin, holder):
    """no_grad SP student dense + top-k selection on the LOCAL head subset.

    Returns (out_dense_h (b,H_local,s_full,d), (k_sel,k_cnt) for local kv heads).
    Selection is over FULL-sequence blocks (each rank holds the whole sequence),
    so it is identical to the single-GPU selection for those heads.
    """
    bsz = hidden.size(0)
    group = holder.sp.group
    H_local = H // holder.sp.sp_size
    with torch.no_grad():
        qf, kf, vf = _sp_project_a2a(self, hidden, H, Hkv, hd, cos, sin, group)
        full_bm = torch.empty((bsz, H_local, 1, 1), dtype=torch.bool,
                              device=qf.device)
        out_dense_h, block_score = block_sparse_attn_with_score(
            qf, kf, vf, full_bm, block_size=holder.block_size,
            segment_size=holder.segment_size, causal=holder.causal)
        k_sel, k_cnt = select_topk_blocks_per_kv_head(
            block_score, G, holder.budget, sink_blocks=holder.sink_blocks,
            local_blocks=holder.local_blocks, causal=holder.causal,
            select_mode=holder.select_mode, top_p=holder.top_p,
            min_blocks=holder.min_blocks, max_blocks=holder.max_blocks)
        return out_dense_h, (k_sel, k_cnt)


def _sp_student_dense_only(self, hidden, H, Hkv, hd, cos, sin, holder):
    """no_grad SP student dense WITHOUT selection (use pass reads frozen sels)."""
    group = holder.sp.group
    with torch.no_grad():
        qf, kf, vf = _sp_project_a2a(self, hidden, H, Hkv, hd, cos, sin, group)
        return block_sparse_attn_dense_no_score(
            qf, kf, vf, block_size=holder.block_size,
            segment_size=holder.segment_size, causal=holder.causal)


def _sp_student_sparse_blend(self, hidden, out_dense_h, sel_prev,
                             H, Hkv, hd, G, cos, sin, holder):
    """Grad-capable SP sparse branch (local heads/full seq) + gate-slice blend.

    Returns the blended attention output (b, H_local, s_full, d) BEFORE the
    all-to-all-back + o_proj. ``out_dense_h`` is the detached anchor value in the
    same (b, H_local, s_full, d) layout. The gate uses ONLY this rank's kv-head
    slice; gradients for the other slices are produced on their owning ranks and
    assembled by the caller's all-reduce.
    """
    group = holder.sp.group
    qf, kf, vf = _sp_project_a2a(self, hidden, H, Hkv, hd, cos, sin, group)
    k_sel_prev, k_cnt_prev = sel_prev
    k_sel_in = k_sel_prev.permute(0, 2, 1, 3).contiguous()
    k_cnt_in = k_cnt_prev.permute(0, 2, 1).contiguous()
    out_sparse_h = sparse_block_attn_trainable_from_cache(
        qf, kf, vf, k_sel_in, k_cnt_in,
        budget=holder.budget, block_size=holder.block_size, causal=holder.causal)
    lo, hi = holder.sp.head_range(Hkv)  # this rank's kv-head range
    # kuma gate is float32; cast to bf16 attn-output dtype (no-op for l1 bf16).
    g = _kv_head_gate(self, holder, lo, hi).to(out_sparse_h.dtype).repeat_interleave(G).view(1, -1, 1, 1)
    return (1 - g) * out_sparse_h + g * out_dense_h  # out_dense_h detached


def llama_reuse_v1_forward_two_way(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """Block-sparse-reuse attention forward with per-phase dispatch (tfm 5.6.0).

    The decoder layer passes ``position_embeddings=(cos, sin)`` and expects a
    2-tuple ``(attn_output, attn_weights)`` return. ``holder.phase`` selects the
    forward mode:
      * ``"single"`` (default): legacy one-pass training. Input is the duplicated
        ``[teacher | student]`` (b=2) stack; grad flows through the sparse branch
        in this same forward.
      * ``"fill"`` / ``"use"``: the original ``--two_pass`` scheme. ``fill`` is a
        no_grad b=2 ``[teacher | student]`` pass; ``use`` is a grad b=1
        student-only pass. Kept for callers that can afford the b=2 fill.
      * ``"fill_teacher"`` / ``"fill_student"`` / ``"use"`` (Option A): the fill
        pass is split into two INDEPENDENT b=1 forwards -- ``fill_teacher`` (dense
        distill target, no selection state) then ``fill_student`` (records every
        layer's top-k selection, resets at L0) -- so the fill-pass MLP transient
        runs at b=1 instead of b=2, halving activation memory (the 128k OOM fix).
        Only ``use`` builds an autograd graph, so the memory peak is a single b=1
        graph and 128k fits without sequence parallelism.
    """
    holder = self._reuse_holder
    phase = getattr(holder, "phase", "single")
    L = self._reuse_layer_idx

    cfg = self.config
    H = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    G = self.num_key_value_groups
    hd = self.head_dim
    hidden_size = cfg.hidden_size
    cos, sin = position_embeddings

    # ---- Two-pass phases. The fill pass is no_grad (b=2 teacher|student, no AC
    # boundaries), so the only grad-carrying / activation-checkpointed pass is
    # "use" at b=1 -> the global memory peak is set by a single b=1 graph, which
    # lets 128k fit without sequence parallelism.
    if phase == "fill_teacher":
        # Option A fill, part 1 (no_grad, b=1): teacher-only dense pass. Produces
        # the distill target. Reads / writes NO selection state, so it can run as
        # a fully independent b=1 forward before the student fill -- halving the
        # fill-pass activation transient (the b=2 MLP was the 128k OOM site).
        if holder.sp.sp_size > 1:
            return _sp_teacher_dense(
                self, hidden_states, H, Hkv, hd, cos, sin, holder, hidden_size), None
        return _teacher_dense(
            self, hidden_states, H, Hkv, hd, cos, sin, holder, hidden_size), None

    if phase == "fill_student":
        # Option A fill, part 2 (no_grad, b=1): student-only pass. Records every
        # layer's top-k selection into holder.sels and reproduces the blend
        # trajectory, so the next layer's selection is computed on exactly the
        # input the use pass will see. Reset lives HERE (not in fill_teacher):
        # the student stream is the one that writes sels.
        #
        # kuma mode: also maintains holder.anchor_cache (per-kv-head IndexCache
        # proxy). After computing this layer's sel = (k_sel, k_cnt), we check the
        # sampled gate z (from frozen holder.kuma_noise[L]) and update
        # anchor_cache[h] only for heads where z[h] > 0.5 (anchor heads). Sparse
        # heads (z[h] <= 0.5) leave their slot unchanged, so they inherit the most
        # recent anchor write -- mirroring inference IndexCache semantics exactly.
        # Layer 0 is all-anchor and always writes all slots (gate frozen at 1).
        bsz, q_len, _ = hidden_states.size()
        if L == 0:
            holder.reset()
        if holder.sp.sp_size > 1:
            out_dense_h, sel = _sp_student_dense_and_select(
                self, hidden_states, H, Hkv, hd, G, cos, sin, holder)
            holder.sels[L] = sel
            group = holder.sp.group
            if L == 0:
                # layer 0 all-anchor: write all local kv-head slots THEN output dense.
                if holder.reg_mode == "kuma":
                    lo, hi = holder.sp.head_range(Hkv)
                    k_sel_l, k_cnt_l = sel  # (b, Hkv_local, nqb, max_sel/nqb)
                    for h_local in range(k_sel_l.shape[1]):
                        h_global = lo + h_local
                        holder.anchor_cache[h_global] = (
                            k_sel_l[:, h_local:h_local+1],
                            k_cnt_l[:, h_local:h_local+1],
                        )
                student_out = _sp_finish(self, out_dense_h, bsz, hidden_size, group)
            else:
                with torch.no_grad():
                    lo_s, hi_s = holder.sp.head_range(Hkv)
                    # Read anchor_cache BEFORE updating it: each head blends with
                    # the most recent anchor write from a PREVIOUS layer, matching
                    # inference IndexCache semantics (anchor heads also read the old
                    # cache before writing their new selection for later layers).
                    sel_for_blend = _anchor_cache_sel(
                        holder, Hkv, holder.sels[L - 1], L, lo=lo_s, hi=hi_s)
                    blended = _sp_student_sparse_blend(
                        self, hidden_states, out_dense_h, sel_for_blend,
                        H, Hkv, hd, G, cos, sin, holder)
                    student_out = _sp_finish(self, blended, bsz, hidden_size, group)
                    # Update anchor_cache AFTER blend: anchor heads (z>0.5) write
                    # their fresh sel so subsequent layers can read it.
                    if holder.reg_mode == "kuma":
                        z = _kv_head_gate(self, holder, lo_s, hi_s)  # (Hkv_local,)
                        k_sel_l, k_cnt_l = sel
                        for h_local in range(k_sel_l.shape[1]):
                            if z[h_local].item() > 0.5:
                                h_global = lo_s + h_local
                                holder.anchor_cache[h_global] = (
                                    k_sel_l[:, h_local:h_local+1],
                                    k_cnt_l[:, h_local:h_local+1],
                                )
            return student_out, None
        out_dense_s, sel = _student_dense_and_select(
            self, hidden_states, H, Hkv, hd, G, cos, sin, holder)
        holder.sels[L] = sel
        if L == 0:
            # layer 0 all-anchor: write every kv-head slot THEN output dense.
            if holder.reg_mode == "kuma":
                k_sel_l, k_cnt_l = sel  # (b, Hkv, nqb, max_sel)
                for h in range(k_sel_l.shape[1]):
                    holder.anchor_cache[h] = (
                        k_sel_l[:, h:h+1], k_cnt_l[:, h:h+1])
            student_out = self.o_proj(out_dense_s.reshape(bsz, q_len, hidden_size))
        else:
            with torch.no_grad():
                # Read anchor_cache BEFORE updating it: each head blends with the
                # most recent anchor write from a PREVIOUS layer, matching inference
                # IndexCache semantics (anchor heads also read the old cache before
                # writing their new selection for subsequent layers).
                sel_for_blend = _anchor_cache_sel(holder, Hkv, holder.sels[L - 1], L)
                student_out = _student_sparse_blend(
                    self, hidden_states, out_dense_s, sel_for_blend,
                    H, Hkv, hd, G, cos, sin, holder, hidden_size)
                # Update anchor_cache AFTER blend: anchor heads (z>0.5) write their
                # fresh sel so subsequent layers can read it.
                if holder.reg_mode == "kuma":
                    z = _kv_head_gate(self, holder)  # (Hkv,)
                    k_sel_l, k_cnt_l = sel
                    for h in range(k_sel_l.shape[1]):
                        if z[h].item() > 0.5:
                            holder.anchor_cache[h] = (
                                k_sel_l[:, h:h+1], k_cnt_l[:, h:h+1])
        return student_out, None

    if phase == "fill":
        # Pass 1 (no_grad, b=2 [teacher|student]): the teacher half produces the
        # distill target (returned as the front half); the student half records
        # every layer's top-k selection into holder.sels AND reproduces the blend
        # trajectory, so the next layer's selection is computed on exactly the
        # input the use pass will see. All no_grad -> no autograd graph, no AC
        # boundaries, so folding teacher+student into one b=2 forward here is
        # cheap and saves a full-model FSDP all-gather vs a separate teacher pass.
        bsz_x_2, q_len, _ = hidden_states.size()
        bsz = bsz_x_2 // 2
        teacher_hidden = hidden_states[:bsz]
        student_hidden = hidden_states[bsz:]
        if L == 0:
            holder.reset()
        teacher_out = _teacher_dense(
            self, teacher_hidden, H, Hkv, hd, cos, sin, holder, hidden_size)
        out_dense_s, sel = _student_dense_and_select(
            self, student_hidden, H, Hkv, hd, G, cos, sin, holder)
        holder.sels[L] = sel
        if L == 0:
            student_out = self.o_proj(out_dense_s.reshape(bsz, q_len, hidden_size))
        else:
            with torch.no_grad():
                student_out = _student_sparse_blend(
                    self, student_hidden, out_dense_s, holder.sels[L - 1],
                    H, Hkv, hd, G, cos, sin, holder, hidden_size)
        return torch.cat([teacher_out, student_out], dim=0), None

    if phase == "use":
        # Pass 2 (grad, student-only): read the FROZEN selection from the fill
        # pass. NO reset, NO write -> the per-layer forward is a pure function of
        # its inputs, so it is recompute-safe under activation checkpointing.
        #
        # kuma mode: read from holder.anchor_cache (per-kv-head, mirrors inference
        # IndexCache) instead of holder.sels[L-1]. anchor_cache was populated by
        # fill_student, so it is frozen here -- same AC-safe guarantee.
        bsz, q_len, _ = hidden_states.size()
        if holder.sp.sp_size > 1:
            group = holder.sp.group
            out_dense_h = _sp_student_dense_only(
                self, hidden_states, H, Hkv, hd, cos, sin, holder)
            if L == 0:
                student_out = _sp_finish(self, out_dense_h, bsz, hidden_size, group)
            else:
                lo_u, hi_u = holder.sp.head_range(Hkv)
                sel_for_blend = _anchor_cache_sel(
                    holder, Hkv, holder.sels[L - 1], L, lo=lo_u, hi=hi_u)
                blended = _sp_student_sparse_blend(
                    self, hidden_states, out_dense_h, sel_for_blend,
                    H, Hkv, hd, G, cos, sin, holder)
                student_out = _sp_finish(self, blended, bsz, hidden_size, group)
            return student_out, None
        out_dense_s = _student_dense_only(
            self, hidden_states, H, Hkv, hd, cos, sin, holder)
        if L == 0:
            student_out = self.o_proj(out_dense_s.reshape(bsz, q_len, hidden_size))
        else:
            sel_for_blend = _anchor_cache_sel(holder, Hkv, holder.sels[L - 1], L)
            student_out = _student_sparse_blend(
                self, hidden_states, out_dense_s, sel_for_blend,
                H, Hkv, hd, G, cos, sin, holder, hidden_size)
        return student_out, None

    # ---- Legacy single-pass b=2 two-way forward (short sequences / no --two_pass).
    bsz_x_2, q_len, _ = hidden_states.size()
    assert bsz_x_2 % 2 == 0, "single-pass forward expects a duplicated (teacher|student) batch"
    bsz = bsz_x_2 // 2
    teacher_hidden = hidden_states[:bsz]
    student_hidden = hidden_states[bsz:]

    if L == 0:
        holder.reset()
    teacher_out = _teacher_dense(
        self, teacher_hidden, H, Hkv, hd, cos, sin, holder, hidden_size)
    out_dense_s, sel = _student_dense_and_select(
        self, student_hidden, H, Hkv, hd, G, cos, sin, holder)
    holder.sels[L] = sel
    if L == 0:
        # Layer 0 is forced all-anchor: student == dense, no sparse/gate branch.
        student_out = self.o_proj(out_dense_s.reshape(bsz, q_len, hidden_size))
    else:
        student_out = _student_sparse_blend(
            self, student_hidden, out_dense_s, holder.sels[L - 1],
            H, Hkv, hd, G, cos, sin, holder, hidden_size)

    return torch.cat([teacher_out, student_out], dim=0), None


def enable_llama_reuse_v1_training(
    model: LlamaForCausalLM,
    budget=32,
    block_size=128,
    segment_size=2048,
    sink_blocks=1,
    local_blocks=2,
    initial_value=1.0,
    causal=True,
    reg_mode="l1",
    select_mode="topk",
    top_p=0.9,
    min_blocks=8,
    max_blocks=64,
):
    """Install the two-way reuse_v1 training forward + per-kv-head gate.

    * ``budget=32`` -> ``max_sel=36`` (topk mode). budget/block_size/sink/local
      MUST match the reuse_v1 inference config and are baked into the exported
      label metadata.
    * One ``ReuseV1TrainHolder`` is shared across all layers so the fresh top-k
      selection computed by layer L's dense stream is visible to layer L+1's
      sparse stream. It resets when layer 0 runs (new-forward signal).
    * ``reg_mode="l1"`` (default): each layer gets a trainable ``anchor_heads``
      gate ``(num_key_value_heads,)`` init ``initial_value``, used directly
      (clamped) in the blend + L1 sparsity.
    * ``reg_mode="kuma"``: each layer ALSO gets trainable HardKuma shape params
      ``kuma_a`` / ``kuma_b`` ``(num_key_value_heads,)`` init 1.0 (neutral); the
      blend gate is the HardKuma reparam ``z`` (see ``_kv_head_gate``) and the
      deterministic ``anchor_heads`` gate is frozen (unused in the blend, kept
      only so the l1 code paths / export helpers stay uniform). Density is driven
      by a Lagrangian constraint in the train loop instead of L1.
    * ``reg_mode="ddp"``: ``anchor_heads`` IS the learnable logit ``z_loga``; the
      blend gate is a straight-through ``ste_clamp(z_loga, 0, 1)`` (see
      ``_kv_head_gate``). No extra params are registered. The trainer noise-inits
      the logit (layer >=1) and floors it per-step (no upper clamp); density is
      driven by a 3-term learnable-lambda Lagrangian on the annealed
      soft-saturation score instead of L1.
    * Layer 0's gate(s) are frozen at all-anchor and excluded from optimization.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    holder = ReuseV1TrainHolder(
        budget=budget,
        block_size=block_size,
        segment_size=segment_size,
        sink_blocks=sink_blocks,
        local_blocks=local_blocks,
        causal=causal,
        select_mode=select_mode,
        top_p=top_p,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
    )
    holder.reg_mode = reg_mode
    holder.stg_sigma = 0.5  # default; overridden in train loop via args.stg_sigma
    holder.hln_sigma = 0.5  # default; overridden in train loop (s value for logging)
    holder.hl_s_raw  = None # set to nn.Parameter by train loop for hln mode
    model._reuse_holder = holder

    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.forward = types.MethodType(llama_reuse_v1_forward_two_way, module)
        module._reuse_holder = holder
        module._reuse_layer_idx = idx

        Hkv = module.config.num_key_value_heads
        gate = nn.Parameter(
            torch.ones(Hkv, device=device, dtype=dtype) * initial_value
        )
        if idx == 0:
            # Layer 0 is forced all-anchor: freeze gate at 1, no gradient, no L1.
            gate.data.fill_(1.0)
            gate.requires_grad_(False)
        module.register_parameter("anchor_heads", gate)

        if reg_mode == "kuma":
            # HardKuma shape params (a, b) init 1.0 -> a base Kuma(1,1) = Uniform,
            # HardKuma density ~ mean gate; the Lagrangian constraint moves them
            # toward desired_density. The deterministic anchor_heads gate is
            # unused in the kuma blend, so freeze it (it stays only for uniform
            # l1/export bookkeeping; export uses hardkuma_mean, not anchor_heads).
            gate.requires_grad_(False)
            kuma_a = nn.Parameter(torch.ones(Hkv, device=device, dtype=dtype))
            kuma_b = nn.Parameter(torch.ones(Hkv, device=device, dtype=dtype))
            if idx == 0:
                kuma_a.requires_grad_(False)
                kuma_b.requires_grad_(False)
            module.register_parameter("kuma_a", kuma_a)
            module.register_parameter("kuma_b", kuma_b)


def enable_sequence_parallel(model, group):
    """Turn on Ulysses sequence parallelism for a reuse_v1-training model.

    ``group`` is the sequence-parallel process group (all ranks that split one
    sequence). Every layer shares the model's holder, so setting ``holder.sp``
    once enables the SP code path in every attention forward. ``group=None``
    (or a size-1 group) leaves SP off. Requires ``num_attention_heads`` and
    ``num_key_value_heads`` both divisible by the group size.
    """
    holder = model._reuse_holder
    holder.sp = SPContext(group)
    if holder.sp.sp_size > 1:
        cfg = model.config
        assert cfg.num_attention_heads % holder.sp.sp_size == 0, (
            f"num_attention_heads={cfg.num_attention_heads} not divisible by "
            f"sp_size={holder.sp.sp_size}")
        assert cfg.num_key_value_heads % holder.sp.sp_size == 0, (
            f"num_key_value_heads={cfg.num_key_value_heads} not divisible by "
            f"sp_size={holder.sp.sp_size}")
    return holder.sp


def _iter_reuse_modules(model):
    """Yield (layer_idx, self_attn_module) for every gated layer, in order.

    Traverses ``model.modules()`` and matches any module carrying both an
    ``anchor_heads`` gate and a ``_reuse_layer_idx``, so it is agnostic to how
    the decoder layers are wrapped -- bare LlamaModel/LlamaForCausalLM, FSDP2,
    per-layer activation-checkpoint wrappers, or grouped-layer segments
    (_GroupedDecoderLayers). Ordering is by ``_reuse_layer_idx`` so the returned
    sequence always matches the true layer order regardless of wrapping.
    """
    found = []
    for module in model.modules():
        if hasattr(module, "anchor_heads") and hasattr(module, "_reuse_layer_idx"):
            found.append((module._reuse_layer_idx, module))
    found.sort(key=lambda x: x[0])
    for idx, module in found:
        yield idx, module


def get_llama_anchor_heads(model):
    """Return a list (len = num_layers) of per-kv-head gate tensors."""
    return [module.anchor_heads for _, module in _iter_reuse_modules(model)]


def get_llama_kuma_params(model):
    """Return ``(a_list, b_list)`` of per-layer HardKuma shape params (kuma mode).

    Each list has length num_layers; entry L is the ``(num_key_value_heads,)``
    parameter for layer L. Layer 0's params are frozen (requires_grad False).
    Raises AttributeError if the model was not enabled with reg_mode='kuma'.
    """
    a_list, b_list = [], []
    for _, module in _iter_reuse_modules(model):
        a_list.append(module.kuma_a)
        b_list.append(module.kuma_b)
    return a_list, b_list


def set_llama_anchor_heads(model, anchor_heads):
    for idx, module in _iter_reuse_modules(model):
        module.anchor_heads.data = anchor_heads[idx].to(
            module.anchor_heads.device, module.anchor_heads.dtype
        )


def map_llama_anchor_heads(model, func):
    for _, module in _iter_reuse_modules(model):
        func(module.anchor_heads)

