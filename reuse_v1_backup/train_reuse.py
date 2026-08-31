"""Train reuse_v1 anchor/sparse head labels, DuoAttention-style.

Learns a per-(layer, kv-head) soft gate ``anchor_heads`` in [0,1] via two-way
distillation (teacher = dense block attention, student = gate-blended
dense/sparse) + L1 sparsity. At convergence, ``gate > 0.5`` -> anchor head;
otherwise sparse. Layer 0 is forced all-anchor (gate frozen at 1, no L1).

Ulysses sequence parallelism (``--sp_size``, default 8) + FSDP2, both on the
full world mesh: one sequence is split across sp_size ranks (each holds
seq_len/sp_size tokens); attention all-to-alls internally so every rank still
sees the WHOLE sequence for its head subset -> global top-k block selection
stays EXACT. dp_size = world_size // sp_size runs distinct samples. sp_size=1
falls back to pure FSDP2 data parallelism (each rank runs the full sequence).

``--two_pass`` splits each step into three forwards: (1) a no_grad b=1 teacher
"fill" pass (dense block attention; the distill target), (2) a no_grad b=1
student "fill" pass (gate-blended; populates holder.sels top-k for all layers),
and (3) a grad b=1 "use" pass that only READS the frozen selections. Because
the fill passes are no_grad (no graph, no activation memory), the only
grad-carrying pass is the use pass, so activation checkpointing (ON, applied in
main) only kicks in there. Combined with SP, 128k fits at ~12.8GB/rank with NO
CPU offload. Two-pass is gradient-exact vs the single-pass forward (top-k
selection is detached either way) at the cost of the extra fill forwards.

The reuse_v1 hyperparameters below are LOCKED and written into the exported
label metadata; they MUST match the reuse_v1 inference config.

Launch (in the project venv, on 8xH800):
    torchrun --nnodes 1 --nproc_per_node 8 reuse_v1/train_reuse.py \
        --model_name <llama> --dataset_name <jsonl> \
        --output_dir outputs/reuse_v1_label --num_steps 2000 \
        --reg_weight 0.05 --lr 0.02 --max_length 131072 --batch_size 1 \
        --two_pass --sp_size 8
    (or just: bash scripts/train_reuse.sh <model> 8192 131072 0.05 0.02 10)
"""

import os
import sys
import json

import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pbs-attn_h
_PATCH_DIR = os.path.join(_REPO_ROOT, "reuse_v1", "patch")
for _p in (_REPO_ROOT, _PATCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reuse_v1.utils import (
    parse_args,
    get_tokenizer,
    visualize_pruned_attention_heads,
    full_attention_heads_to_list,
    save_full_attention_heads,
    seed_everything,
)
from reuse_v1.data import (
    get_dataset,
    MultiplePasskeyRetrievalDataset,
    get_supervised_dataloader,
)
from reuse_v1.loss import l1_loss

# The training patch is imported as a standalone module (not via the heavy
# reuse_v1.patch package __init__, which pulls in flash_attn / tensor_parallel).
from reuse_train_llama import (
    enable_llama_reuse_v1_training,
    enable_sequence_parallel,
    get_llama_anchor_heads,
    get_llama_kuma_params,
    set_llama_anchor_heads,
    map_llama_anchor_heads,
)
# HardKuma density (Lagrangian target) + mean (export statistic) for --reg_mode kuma.
from kuma_gate import hardkuma_density, hardkuma_mean, NOISE_EPS
# DDP (deterministic differentiable pruning) gate helpers for --reg_mode ddp:
# annealed soft-saturation score (Lagrangian readout) + deterministic export mask.
from ddp_gate import soft_saturation_score, anneal_mean, deterministic_head_mask
from stg_gate import stg_density_mean
from hc_gate import hc_density, hc_p_nonzero, hc_p_positive, NOISE_EPS as HC_NOISE_EPS
from hln_gate import hl_density_mean, hl_s_from_raw, HL_S_MIN, NOISE_EPS as HL_NOISE_EPS

import torch.nn as nn

import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ---- LOCKED reuse_v1 hyperparameters (MUST match inference) ----
BUDGET = 32
BLOCK_SIZE = 128
SEGMENT_SIZE = 2048
SINK_BLOCKS = 1
LOCAL_BLOCKS = 2
SELECT_MODE = "topk"


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


class _GroupedDecoderLayers(torch.nn.Module):
    """Run ``group_size`` consecutive LlamaDecoderLayers as ONE unit.

    Used only for grouped activation checkpointing (--ac_group_size > 1): one
    checkpoint segment wraps this whole group, so backward saves a single
    boundary tensor per group (the group's input) instead of one per layer.
    The inner LlamaDecoderLayer objects (and their patched self_attn /
    _reuse_layer_idx) are preserved unchanged, so FSDP2 sharding and the
    cross-layer holder.sels threading keep working. tfm 5.6.0 decoder layers
    return a bare tensor and take everything after hidden_states as kwargs.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states


def _group_decoder_layers(model, group_size):
    """Repack model.layers into ceil(N/group_size) _GroupedDecoderLayers.

    LlamaModel.forward iterates ``self.layers[: num_hidden_layers]`` generically
    (no absolute indexing) and slicing a shorter list is a no-op, so the grouped
    ModuleList drops in transparently.
    """
    src = list(model.layers)
    grouped = [
        _GroupedDecoderLayers(src[i:i + group_size])
        for i in range(0, len(src), group_size)
    ]
    model.layers = torch.nn.ModuleList(grouped)


def apply_fsdp(model, mesh, mp_policy, modules_to_shard):
    """FSDP2 data parallelism. reshard_after_forward True to save memory."""
    fsdp_config = {"mp_policy": mp_policy, "mesh": mesh, "reshard_after_forward": True}
    for module in model.modules():
        if any(isinstance(module, m) for m in modules_to_shard):
            fully_shard(module, **fsdp_config)
    fully_shard(model, **fsdp_config)


def _stack_gates(model):
    """Gather per-layer (Hkv,) gates into a list of full (unsharded) tensors."""
    gates = get_llama_anchor_heads(model)
    return [g.full_tensor() if hasattr(g, "full_tensor") else g for g in gates]


def _stack_kuma(model):
    """Gather per-layer HardKuma (a, b) into two lists of full (unsharded) tensors.

    Like ``_stack_gates`` but for the ``--reg_mode kuma`` shape params. Under
    FSDP2 the (Hkv,) params are sharded, so ``full_tensor()`` all-gathers them;
    its backward reduce-scatters the gradient across ranks (SUM), matching the
    ``_stack_gates`` L1 convention. The adaptive Lagrange multiplier absorbs any
    constant world-size scaling, so the density constraint stays well-posed.
    """
    a_list, b_list = get_llama_kuma_params(model)
    a_full = [a.full_tensor() if hasattr(a, "full_tensor") else a for a in a_list]
    b_full = [b.full_tensor() if hasattr(b, "full_tensor") else b for b in b_list]
    return a_full, b_full


def _make_sp_group(sp_size, world_size, rank):
    """Build this rank's Ulysses sequence-parallel process group.

    Ranks are laid out row-major: SP group ``d`` owns the contiguous rank block
    ``[d*sp_size, (d+1)*sp_size)`` and all its members split ONE sequence; the
    ``dp_size = world_size // sp_size`` groups run distinct samples. FSDP2 still
    shards weights across the FULL flat world mesh, so the ``* world_size`` loss
    recipe (see train()) reduces gate grads correctly over every rank: SP
    contributions to one sample SUM (compensating FSDP's average via * world),
    and distinct dp samples SUM into a global mean-over-all-labels gradient.

    ``dist.new_group`` is collective, so EVERY rank walks the full dp loop and
    creates all groups in lockstep, keeping only the one it belongs to. Returns
    (sp_group, sp_size, sp_rank, dp_size, dp_rank). sp_size<=1 -> (None,1,0,...).
    """
    if sp_size <= 1:
        return None, 1, 0, world_size, rank
    assert world_size % sp_size == 0, (
        f"world_size={world_size} not divisible by sp_size={sp_size}")
    dp_size = world_size // sp_size
    dp_rank = rank // sp_size
    sp_rank = rank % sp_size
    sp_group = None
    for d in range(dp_size):
        ranks = list(range(d * sp_size, (d + 1) * sp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            sp_group = g
    return sp_group, sp_size, sp_rank, dp_size, dp_rank


def train(args, model, rank, world_size, train_dataloader, optimizer, scheduler,
          resume_step, pad_token_id, lambdas=None):
    model.train()
    if rank == 0:
        pbar = tqdm(range(args.num_steps))
    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    reg_mode = getattr(model._reuse_holder, "reg_mode", "l1")
    # --- Lagrangian density-constraint state (--reg_mode kuma / stg) ---
    # Mirrors the reference train_kuma: an adaptive multiplier lambda0 chases the
    # density constraint density(a,b) <= desired_density; c0_ma is a slow moving
    # average of the (relu'd) constraint violation. These persist across steps.
    if reg_mode == "kuma":
        desired_density = args.desired_density
        c0_ma = torch.tensor(0.0)
        lambda0 = torch.tensor(float(args.lamda_init_value))
        lagrange_alpha = 0.9
        lagrange_lr = args.lagrange_lr
        lambda_min = 1e-12
        lambda_max = 20.0
        # Per-kv-head gate count for the frozen per-step reparam noise.
        _gates0 = get_llama_anchor_heads(model)
        num_layers_kuma = len(_gates0)
        hkv_kuma = _gates0[0].shape[0]
    elif reg_mode == "stg":
        # STG: same Lagrangian structure as kuma but with Gaussian noise and
        # the simpler Phi-based density readout. desired_density = 1 - target_sparsity.
        desired_density = 1.0 - args.target_sparsity
        c0_ma = torch.tensor(0.0)
        lambda0 = torch.tensor(float(getattr(args, "lambda_init_value", 1.0)))
        lagrange_alpha = 0.9
        lagrange_lr = getattr(args, "lagrange_lr", args.lambda_1_lr)
        lambda_min = 1e-12
        lambda_max = 20.0
        stg_sigma = getattr(args, "stg_sigma", 0.5)
        _gates0 = get_llama_anchor_heads(model)
        num_layers_stg = len(_gates0)
        hkv_stg = _gates0[0].shape[0]
    elif reg_mode == "hln":
        # HardLogistic(mu, s): HC with learnable global temperature s.
        # density = sigmoid(mu), s-free (s cancels at z>0.5 threshold).
        # Lagrangian structure identical to stg/kuma.
        # hl_s_raw is created in main() BEFORE the optimizer so it is in
        # param_groups; here we just read it from holder for sampling.
        desired_density = 1.0 - args.target_sparsity
        c0_ma = torch.tensor(0.0)
        lambda0 = torch.tensor(float(getattr(args, "lambda_init_value", 1.0)))
        lagrange_alpha = 0.9
        lagrange_lr = getattr(args, "lagrange_lr", args.lambda_1_lr)
        lambda_min = 1e-12
        lambda_max = 20.0
        hl_s_raw = model._reuse_holder.hl_s_raw   # nn.Parameter from main()
        _gates0 = get_llama_anchor_heads(model)
        num_layers_hln = len(_gates0)
        hkv_hln = _gates0[0].shape[0]
    elif reg_mode == "hc":
        # Hard Concrete: original Louizos 2018 formulation.
        # reg_loss = reg_weight * sum P(z_j != 0) = reg_weight * sum sigmoid(alpha+1.599)
        # Fixed coefficient reg_weight (from --reg_weight, same as l1 mode).
        # No Lagrangian. Sparsity at convergence is controlled by reg_weight:
        #   larger reg_weight → more heads pushed to alpha<<0 → more sparse.
        # Export: alpha > 0  <=>  P(z>0) > 0.5  (natural threshold, aligns with
        # inference label > 0.5).  The two-polar convergence (anchor heads alpha>>0,
        # sparse heads alpha<<0) is driven by the interplay of distill_loss pulling
        # important heads up and the L0 penalty pushing all heads down.
        # desired_density / Lagrangian state not used in this mode.
        _gates0 = get_llama_anchor_heads(model)
        num_layers_hc = len(_gates0)
        hkv_hc = _gates0[0].shape[0]
    elif reg_mode == "hcl":
        # Hard Concrete + Lagrangian (hcl):
        # Same HC gate as 'hc' (anchor_heads IS log_alpha, z ~ HC(log_alpha)).
        # density = mean(sigmoid(log_alpha)) = mean(P(z>0.5)), aligned with
        # export threshold log_alpha>0 (both are the same function of log_alpha).
        # Adaptive single-lambda Lagrangian drives density toward 1-target_sparsity.
        # This avoids the L0 export gap: sigmoid(alpha) and alpha>0 are directly
        # aligned (no +1.599 offset), so constraint value = export fraction at
        # convergence (when log_alpha is fully polarized to +/-inf).
        desired_density = 1.0 - args.target_sparsity
        c0_ma = torch.tensor(0.0)
        lambda0 = torch.tensor(float(getattr(args, "lambda_init_value", 1.0)))
        lagrange_alpha = 0.9
        lagrange_lr = getattr(args, "lagrange_lr", args.lambda_1_lr)
        lambda_min = 1e-12
        lambda_max = 20.0
        _gates0 = get_llama_anchor_heads(model)
        num_layers_hcl = len(_gates0)
        hkv_hcl = _gates0[0].shape[0]

    while True:
        if global_step >= args.num_steps:
            break
        for batch in train_dataloader:
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                continue

            # Per-step gate clamp. l1/kuma/stg: the anchor_heads gate is a [0,1]
            # blend weight, so pin it to [0,1]. ddp: anchor_heads is the raw
            # LOGIT z_loga -- clamp to [z_loga_clamp_min, 1.1] to keep
            # soft_saturation_score in its effective gradient region. Without
            # the upper bound z_loga can grow unbounded, saturating the
            # soft-saturation sigmoid to 1 (zero gradient) and making the
            # blend gate ste_clamp permanently 1 (distill loss = 0), causing
            # the Lagrangian constraint to stall.
            # stg: mu_code is unconstrained in principle (Phi is defined for all R),
            # but clamp to [-3, 3] to prevent numerical overflow in erf and to
            # keep z = clip(mu+0.5+sigma*eps) in its effective gradient region.
            if reg_mode == "ddp":
                _zmin = args.z_loga_clamp_min
                _zmax = 1.1  # matches LIMIT_B in ddp_gate.py (stretch support upper)

                @torch.no_grad()
                def clamp_(x):
                    x.clamp_(min=_zmin, max=_zmax)
            elif reg_mode == "stg":
                @torch.no_grad()
                def clamp_(x):
                    x.clamp_(-3.0, 3.0)
            elif reg_mode == "hc":
                # log_alpha unconstrained in theory; clamp to prevent overflow
                # in sigmoid (sigmoid(±10) ≈ 1/0 to machine precision).
                @torch.no_grad()
                def clamp_(x):
                    x.clamp_(-10.0, 10.0)
            elif reg_mode in ("hln", "hcl"):
                # hln: mu unconstrained; hcl: log_alpha unconstrained.
                # Both clamp same range as hc.
                @torch.no_grad()
                def clamp_(x):
                    x.clamp_(-10.0, 10.0)
            else:
                @torch.no_grad()
                def clamp_(x):
                    x.clamp_(0, 1)

            map_llama_anchor_heads(model, clamp_)

            batch = {k: v.to(f"cuda:{local_rank}") for k, v in batch.items()}

            holder = model._reuse_holder
            if reg_mode == "kuma":
                # Freeze the HardKuma reparam NOISE for the WHOLE step: drawn ONCE
                # here (outside every forward / AC region) with a rank-shared seed
                # so (1) the fill and use passes + any AC recompute reproduce the
                # SAME z, and (2) all ranks agree on u (SP heads owned by distinct
                # ranks, and the density term is u-independent anyway). z itself is
                # recomputed INSIDE each layer forward from self.kuma_a/kuma_b so
                # the (a,b) gradient stays in the FSDP-hooked forward. Shape
                # (num_layers, Hkv); clamped off {0,1} like the reference sampler.
                gen = torch.Generator(device=f"cuda:{local_rank}")
                gen.manual_seed(args.seed + global_step)
                u = torch.rand(
                    num_layers_kuma, hkv_kuma,
                    generator=gen, device=f"cuda:{local_rank}",
                    dtype=torch.float32,
                ).clamp_(NOISE_EPS, 1.0 - NOISE_EPS)
                holder.kuma_noise = u
            elif reg_mode == "stg":
                # Freeze Gaussian noise for the WHOLE step (same pattern as kuma):
                # drawn once with a rank-shared seed so fill + use passes + AC
                # recompute all see the same epsilon. z is recomputed inside each
                # layer forward from self.anchor_heads (mu_code) + frozen noise.
                gen = torch.Generator(device=f"cuda:{local_rank}")
                gen.manual_seed(args.seed + global_step)
                eps = torch.randn(
                    num_layers_stg, hkv_stg,
                    generator=gen, device=f"cuda:{local_rank}",
                    dtype=torch.float32,
                )
                holder.stg_noise = eps
            elif reg_mode == "hln":
                # HardLogistic(mu, s): uniform noise, same protocol as hc.
                gen = torch.Generator(device=f"cuda:{local_rank}")
                gen.manual_seed(args.seed + global_step)
                u = torch.rand(
                    num_layers_hln, hkv_hln,
                    generator=gen, device=f"cuda:{local_rank}",
                    dtype=torch.float32,
                ).clamp_(HL_NOISE_EPS, 1.0 - HL_NOISE_EPS)
                holder.hln_noise = u
            elif reg_mode in ("hc", "hcl"):
                # Freeze uniform noise for the WHOLE step (same pattern as kuma).
                # hc and hcl share the same HC gate sampling (hc_sample_z + hc_noise).
                gen = torch.Generator(device=f"cuda:{local_rank}")
                gen.manual_seed(args.seed + global_step)
                u = torch.rand(
                    num_layers_hc if reg_mode == "hc" else num_layers_hcl,
                    hkv_hc if reg_mode == "hc" else hkv_hcl,
                    generator=gen, device=f"cuda:{local_rank}",
                    dtype=torch.float32,
                ).clamp_(HC_NOISE_EPS, 1.0 - HC_NOISE_EPS)
                holder.hc_noise = u
            if getattr(args, "two_pass", False):
                # Two-pass, Option A: the fill pass is split into two INDEPENDENT
                # b=1 no_grad forwards so the fill-pass MLP transient stays at b=1
                # (the b=2 fold OOM'd at 128k in down_proj). fill_teacher produces
                # the distill target (reads/writes no selection); fill_student
                # records every layer's top-k selection into holder.sels and
                # reproduces the blend trajectory the use pass will see. Gates are
                # clamped+frozen for the whole step, so these selections match
                # exactly what the use pass reproduces. Neither builds a graph.
                #
                # Under Ulysses SP (holder.sp.sp_size > 1) each rank owns a
                # CONTIGUOUS sequence shard (global tokens [lo, hi)); attention
                # all-to-alls internally so every rank still sees the whole
                # sequence for its head subset. We pad the full sequence to a
                # multiple of lcm(block_size, sp_size) so both the shard boundary
                # (seq % sp_size == 0) and the block-sparse kernel (s_full %
                # block_size == 0) are satisfied; pad tokens are right-appended
                # with label -100 (excluded from distill) and, being causal-after
                # the real tokens, do not perturb any real hidden state.
                sp = holder.sp
                full_ids = batch["input_ids"]
                full_labels = batch["labels"]
                if sp.sp_size > 1:
                    pad_mult = (BLOCK_SIZE if BLOCK_SIZE % sp.sp_size == 0
                                else BLOCK_SIZE * sp.sp_size)
                    seq_full = full_ids.shape[1]
                    pad = (-seq_full) % pad_mult
                    if pad:
                        b = full_ids.shape[0]
                        full_ids = torch.cat(
                            [full_ids, full_ids.new_full((b, pad), pad_token_id)],
                            dim=1)
                        full_labels = torch.cat(
                            [full_labels, full_labels.new_full((b, pad), -100)],
                            dim=1)
                        seq_full += pad
                    s_local = seq_full // sp.sp_size
                    lo = sp.sp_rank * s_local
                    hi = lo + s_local
                    ids_local = full_ids[:, lo:hi].contiguous()
                    labels_local = full_labels[:, lo:hi].contiguous()
                    position_ids = torch.arange(
                        lo, hi, device=ids_local.device).unsqueeze(0)
                    seq_len = seq_full
                else:
                    ids_local = full_ids
                    labels_local = full_labels
                    seq_len = ids_local.shape[1]
                    position_ids = torch.arange(
                        seq_len, device=ids_local.device).unsqueeze(0)

                holder.phase = "fill_teacher"
                with torch.no_grad():
                    teacher_out = model(input_ids=ids_local,
                                        position_ids=position_ids)
                teacher_h = teacher_out[0]

                holder.phase = "fill_student"
                with torch.no_grad():
                    model(input_ids=ids_local,
                          position_ids=position_ids)

                # Pass 2 (grad, b=1 student): read the frozen selections;
                # recompute-safe, so activation checkpointing (applied in main)
                # kicks in here. Only this pass builds a graph.
                holder.phase = "use"
                student_out = model(input_ids=ids_local,
                                    position_ids=position_ids)
                student_h = student_out[0]
                labels = labels_local
            else:
                # Single-pass: duplicate the batch for the two-way (teacher|
                # student) forward; grad flows through the sparse branch here.
                input_ids = torch.cat(
                    [batch["input_ids"], batch["input_ids"]], dim=0)
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(
                    seq_len, device=input_ids.device).unsqueeze(0)
                holder.phase = "single"
                outputs = model(input_ids=input_ids, position_ids=position_ids)
                hidden_states = outputs[0]
                teacher_h = hidden_states[: args.batch_size]
                student_h = hidden_states[args.batch_size :]
                labels = batch["labels"]

            label_mask = labels != -100
            num_labels = label_mask.sum()
            global_num_labels = num_labels.clone().detach()
            dist.all_reduce(global_num_labels)
            global_num_labels = global_num_labels.clamp_min(1)

            teacher_h = teacher_h[label_mask].float()
            student_h = student_h[label_mask].float()

            distill_loss = (
                (teacher_h - student_h).pow(2).mean(dim=-1).sum()
                * world_size
                / global_num_labels
            )

            if reg_mode == "kuma":
                # HardKuma density constraint (adaptive-lambda Lagrangian), the
                # opt-in replacement for L1. density = E[anchor fraction] over
                # ALL layers (layer 0 INCLUDED, matching the reference
                # train_kuma's full-model density scope). Layer 0's (a,b) are
                # frozen at 1.0 -> it contributes a constant ~0.917 to the mean
                # (a=b=1 HardKuma non-zero mass), mirroring the reference where
                # layer 0 ends up dense; no gradient flows to it (frozen leaf),
                # so including it only shifts the density scope, not training.
                # The (a,b) gradient of the GATED layers reaches here via
                # full_tensor() (SUM across ranks, like the L1 path) AND via z
                # inside the forward (FSDP AVERAGE x world); the adaptive lambda
                # absorbs any constant scale so the constraint stays well-posed.
                # c0 uses the smoothed MA value with the instantaneous gradient
                # (reference train_kuma).
                a_full, b_full = _stack_kuma(model)
                a_cat = torch.cat(
                    [a.to(teacher_h.device).clamp(1e-6, 100.0) for a in a_full]
                ).float()
                b_cat = torch.cat(
                    [b.to(teacher_h.device).clamp(1e-6, 100.0) for b in b_full]
                ).float()
                density = hardkuma_density(a_cat, b_cat)
                c0_hat = torch.relu(density - desired_density)
                c0_ma = lagrange_alpha * c0_ma + (1 - lagrange_alpha) * c0_hat.item()
                c0 = c0_hat + (c0_ma.detach() - c0_hat.detach())
                lambda0 = lambda0 * torch.exp(lagrange_lr * c0.detach())
                lambda0 = lambda0.clamp(lambda_min, lambda_max)
                reg_loss = lambda0.detach().to(c0.device) * c0
                loss = distill_loss + reg_loss
                # For logging / export: the HardKuma mean is the anchor decision
                # statistic (mean > 0.5 -> anchor), matching _export_label.
                gates = [
                    hardkuma_mean(a.to(teacher_h.device).clamp(1e-6, 100.0).float(),
                                  b.to(teacher_h.device).clamp(1e-6, 100.0).float())
                    for a, b in zip(a_full, b_full)
                ]
            elif reg_mode == "ddp":
                # Deterministic differentiable pruning: anchor_heads IS the logit
                # z_loga. The 3-term learnable-lambda Lagrangian drives the annealed
                # expected SPARSE fraction (1 - score.mean()) toward target_sparsity.
                # The gate logit reaches here via full_tensor() (SUM backward across
                # ranks, same structure as the validated l1 path) AND via the
                # straight-through gate inside the forward (FSDP AVERAGE x world);
                # the learnable lambdas are plain (replicated) params whose gradient
                # (s - target) is identical on every rank because s comes from the
                # replicated full_tensor score -> they stay in sync with NO manual
                # all-reduce (dual ascent, maximize=True optimizer groups).
                z_full = _stack_gates(model)
                z_full = [g.to(teacher_h.device) for g in z_full]
                z_loga = torch.stack(z_full, dim=0).float()  # (num_layers, Hkv)
                progress = global_step / max(1, args.num_steps)
                anneal_m = anneal_mean(
                    progress,
                    schedule=args.anneal_schedule,
                    mean_min=args.anneal_mean_min,
                    mean_max=0.5,
                    warmup_ratio=args.anneal_warmup_ratio,
                )
                score = soft_saturation_score(z_loga, anneal_m)
                expected_sparsity = 1.0 - score.mean()
                diff = expected_sparsity - args.target_sparsity
                lam1 = lambdas["lambda_1"].to(diff.device)
                lam2 = lambdas["lambda_2"].to(diff.device)
                lag = lam1 * diff + lam2 * diff.pow(2)
                if not args.no_binary_loss:
                    lam3 = lambdas["lambda_3"].to(diff.device)
                    binary = lam3 * ((1.0 - score) * score).mean()
                else:
                    binary = torch.zeros_like(lag)
                reg_loss = (lag + binary).mean()
                loss = distill_loss + reg_loss
                # For logging/export bookkeeping keep the raw per-layer logits
                # (export applies deterministic_head_mask, not a > 0.5 threshold).
                gates = z_full
            elif reg_mode == "stg":
                # STG: anchor_heads IS mu_code. The stochastic gate z is computed
                # from mu + frozen noise inside the layer forward (via holder.stg_noise).
                # Here we compute the Phi-based density for the Lagrangian.
                # Same adaptive-lambda structure as kuma (single lambda0, multiplicative
                # update), but with the simpler Gaussian CDF density readout.
                # mu reaches here via full_tensor() (SUM across ranks, same as l1/ddp).
                mu_full = _stack_gates(model)
                mu_full = [g.to(teacher_h.device) for g in mu_full]
                # Exclude layer 0 (frozen all-anchor, mu=1.0, density≈0.9987) from
                # the Lagrangian density so desired_density = 59/248 directly maps to
                # "59 anchor heads out of 248 trainable (layer>=1) heads", matching
                # the semantics of deterministic_head_mask export (which also skips
                # layer 0). layer 0 still appears in mu_full for logging/export.
                mu_cat = torch.stack(mu_full[1:], dim=0).float()  # (num_layers-1, Hkv)
                density = stg_density_mean(mu_cat, stg_sigma)
                c0_hat = torch.relu(density - desired_density)
                c0_ma = lagrange_alpha * c0_ma + (1 - lagrange_alpha) * c0_hat.item()
                c0 = c0_hat + (c0_ma.detach() - c0_hat.detach())
                lambda0 = lambda0 * torch.exp(lagrange_lr * c0.detach())
                lambda0 = lambda0.clamp(lambda_min, lambda_max)
                reg_loss = lambda0.detach().to(c0.device) * c0
                loss = distill_loss + reg_loss
                # For logging/export: raw mu_full (export uses deterministic_head_mask).
                gates = mu_full
            elif reg_mode == "hln":
                # HardLogistic(mu, s): density = sigmoid(mu), s-free.
                # s is a global learnable temperature (hl_s_raw parameter).
                # Lagrangian structure identical to stg/kuma.
                mu_full = _stack_gates(model)
                mu_full = [g.to(teacher_h.device) for g in mu_full]
                mu_cat = torch.stack(mu_full[1:], dim=0).float()  # (num_layers-1, Hkv)
                density = hl_density_mean(mu_cat)   # sigmoid(mu).mean(), s-free
                c0_hat = torch.relu(density - desired_density)
                c0_ma = lagrange_alpha * c0_ma + (1 - lagrange_alpha) * c0_hat.item()
                c0 = c0_hat + (c0_ma.detach() - c0_hat.detach())
                lambda0 = lambda0 * torch.exp(lagrange_lr * c0.detach())
                lambda0 = lambda0.clamp(lambda_min, lambda_max)
                reg_loss = lambda0.detach().to(c0.device) * c0
                loss = distill_loss + reg_loss
                # Log current s value for monitoring.
                _hl_s = hl_s_from_raw(hl_s_raw).item()
                holder.hln_sigma = _hl_s   # for external logging
                gates = mu_full
            elif reg_mode == "hc":
                # Hard Concrete original formulation (Louizos et al. ICLR 2018):
                # reg_loss = reg_weight * sum_{layer>=1} P(z_h != 0)
                #          = reg_weight * sum sigmoid(log_alpha_h + 1.599)
                # Fixed coefficient reg_weight (from --reg_weight, same as l1 mode).
                # No Lagrangian. Sparsity at convergence is controlled by reg_weight:
                #   larger reg_weight → more heads pushed to alpha<<0 → more sparse.
                # Distill loss pulls important heads to alpha>>0 (z≈1, anchor).
                # L0 penalty pushes all heads toward alpha<<0.
                # Two-polar convergence → alpha>0 is the natural export threshold.
                #
                # To hit target_sparsity=0.79: tune --reg_weight.
                # Rule of thumb: start with reg_weight=0.1, adjust by 2x up/down.
                la_full = _stack_gates(model)
                la_full = [g.to(teacher_h.device) for g in la_full]
                la_cat = torch.cat(la_full[1:]).float()   # layer>=1, all heads flat
                # L0 = sum P(z!=0) = sum sigmoid(alpha + 1.599)
                l0_reg = hc_p_nonzero(la_cat).sum()
                reg_loss = args.reg_weight * l0_reg
                loss = distill_loss + reg_loss
                gates = la_full
            elif reg_mode == "hcl":
                # Hard Concrete + Lagrangian:
                # density = mean(sigmoid(log_alpha)) = mean(P(z>0.5)), aligned
                # with export threshold log_alpha>0 (both use sigmoid(alpha)).
                # Adaptive-lambda Lagrangian drives density toward desired_density.
                la_full = _stack_gates(model)
                la_full = [g.to(teacher_h.device) for g in la_full]
                la_cat = torch.stack(la_full[1:], dim=0).float()  # (num_layers-1, Hkv)
                # density = mean(P(z>0.5)) = mean(sigmoid(alpha)), s-free analog
                density = hc_p_positive(la_cat).mean()
                c0_hat = torch.relu(density - desired_density)
                c0_ma = lagrange_alpha * c0_ma + (1 - lagrange_alpha) * c0_hat.item()
                c0 = c0_hat + (c0_ma.detach() - c0_hat.detach())
                lambda0 = lambda0 * torch.exp(lagrange_lr * c0.detach())
                lambda0 = lambda0.clamp(lambda_min, lambda_max)
                reg_loss = lambda0.detach().to(c0.device) * c0
                loss = distill_loss + reg_loss
                gates = la_full
            else:
                gates = _stack_gates(model)
                gates = [g.to(teacher_h.device) for g in gates]
                # Exclude layer 0 (frozen all-anchor) from the L1 sparsity pressure.
                reg_loss = l1_loss(torch.cat(gates[1:]).float())
                loss = distill_loss + args.reg_weight * reg_loss

            loss.backward()

            local_step = (local_step + 1) % args.gradient_accumulation_steps

            dist.all_reduce(distill_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(reg_loss, op=dist.ReduceOp.AVG)

            if local_step != 0:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Clamp DDP Lagrange multipliers to prevent runaway dual ascent.
            if reg_mode == "ddp" and lambdas is not None:
                with torch.no_grad():
                    if "lambda_1" in lambdas:
                        lambdas["lambda_1"].clamp_(-10, 10)
                    for key in ("lambda_2", "lambda_3"):
                        if key in lambdas:
                            lambdas[key].clamp_(0, 20)

            if rank == 0:
                # hc: snapshot log_alpha before full_attention_heads_to_list
                # converts gate tensors to plain Python lists in-place.
                if reg_mode == "hc":
                    _la_snap = torch.cat([g.detach().float() for g in gates[1:]])
                    _l0_density = (_la_snap > 0).float().mean().item()

                gates_list = full_attention_heads_to_list(gates)

                if not args.disable_wandb:
                    fig = visualize_pruned_attention_heads(gates_list)
                    log_dict = {
                        "distill_loss": distill_loss.item(),
                        "reg_loss": reg_loss.item(),
                        "attn_heads": fig,
                        "step": global_step,
                        "sample_len": seq_len,
                        "num_labels": int(global_num_labels),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    if reg_mode == "kuma":
                        log_dict["density"] = float(density.item())
                        log_dict["lambda0"] = float(lambda0.item())
                    if reg_mode == "stg":
                        log_dict["density"] = float(density.item())
                        log_dict["lambda0"] = float(lambda0.item())
                    if reg_mode == "hln":
                        log_dict["density"] = float(density.item())
                        log_dict["lambda0"] = float(lambda0.item())
                        log_dict["hln_sigma"] = float(holder.hln_sigma)
                    if reg_mode == "hcl":
                        log_dict["density"] = float(density.item())
                        log_dict["lambda0"] = float(lambda0.item())
                    if reg_mode == "hc":
                        log_dict["l0_density"] = _l0_density   # fraction alpha>0
                        log_dict["l0_reg"] = float(reg_loss.item())
                    if reg_mode == "ddp":
                        log_dict["expected_sparsity"] = float(expected_sparsity.item())
                        log_dict["anneal_mean"] = float(anneal_m)
                        log_dict["lambda_1"] = float(lam1.item())
                        log_dict["lambda_2"] = float(lam2.item())
                        if not args.no_binary_loss:
                            log_dict["lambda_3"] = float(lam3.item())
                    wandb.log(log_dict, step=global_step)
                    plt.close(fig)

                if reg_mode == "kuma":
                    extra = (f"|dens={float(density.item()):.3f}"
                             f"|lam={float(lambda0.item()):.2e}")
                elif reg_mode == "stg":
                    extra = (f"|dens={float(density.item()):.3f}"
                             f"|lam={float(lambda0.item()):.2e}")
                elif reg_mode == "hln":
                    extra = (f"|dens={float(density.item()):.3f}"
                             f"|lam={float(lambda0.item()):.2e}"
                             f"|s={float(holder.hln_sigma):.3f}")
                elif reg_mode == "hcl":
                    extra = (f"|dens={float(density.item()):.3f}"
                             f"|lam={float(lambda0.item()):.2e}")
                elif reg_mode == "hc":
                    extra = (f"|l0={_l0_density:.3f}"
                             f"|rw={args.reg_weight:.3f}")
                elif reg_mode == "ddp":
                    extra = (f"|sp={float(expected_sparsity.item()):.3f}"
                             f"|m={float(anneal_m):.3f}"
                             f"|l1={float(lam1.item()):.2e}")
                else:
                    extra = ""
                pbar.set_description(
                    f"Len={seq_len}|N={int(global_num_labels)}"
                    f"|D={distill_loss.item():.3f}|R={reg_loss.item():.3f}"
                    f"{extra}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

                if (args.output_dir is not None
                        and global_step % args.save_steps == 0):
                    _save_checkpoint(args, gates_list, optimizer, scheduler,
                                     global_step, rank)

            if global_step >= args.num_steps:
                break

    if rank == 0:
        pbar.close()


def _save_checkpoint(args, gates_list, optimizer, scheduler, global_step, rank):
    save_full_attention_heads(
        gates_list,
        os.path.join(args.output_dir, f"anchor_heads_step={global_step}.tsv"),
    )
    latest = os.path.join(args.output_dir, "anchor_heads_latest.tsv")
    save_full_attention_heads(gates_list, latest)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        },
        os.path.join(args.output_dir,
                     f"optimizer_scheduler_state_latest-rank={rank}.pt"),
    )


def _export_label(gates, output_dir, reg_mode="l1", desired_density=None,
                  target_sparsity=None, uniform_sparsity=False):
    """Write the reuse_v1 label: (num_layers, Hkv) bool with layer 0 all-anchor.

    Saves ``label.pt`` (bool tensor consumed by reuse_prefill.load_label),
    ``anchor_heads.tsv`` (raw float gates for inspection) and ``config.json``
    with the locked hyperparameters. ``gates`` are the per-layer anchor decision
    statistics: the deterministic gate (l1 mode) or the HardKuma mean (kuma
    mode), thresholded at ``> 0.5``; or the raw logit z_loga (ddp/stg modes),
    where a deterministic top-k head mask hits the EXACT ``target_sparsity``.
    For ``hc`` mode: gates are log_alpha values; when training starts from
    initial_value >> 0, anchor heads converge to log_alpha > 0 and sparse heads
    to log_alpha < 0, so ``log_alpha > 0`` is the natural threshold (P(z>0) >
    0.5) and aligns exactly with the inference ``label > 0.5`` check in
    reuse_prefill.load_label.
    """
    gate_mat = torch.stack([g.detach().float().cpu() for g in gates], dim=0)
    if reg_mode in ("ddp", "stg"):
        # Zero the lowest-scoring (layer>=1) heads to hit the exact target
        # sparse fraction; layer 0 is force-kept anchor inside the helper.
        # For stg: mu_code values are monotone with Phi((mu+0.5)/sigma), so
        # top-k on mu_code gives the same ranking as top-k on the density.
        hard = deterministic_head_mask(
            gate_mat, target_sparsity, uniform_sparsity=uniform_sparsity)
        label = hard > 0.5
    elif reg_mode == "hc":
        # Hard Concrete (Louizos 2018 original): fixed L0 penalty, no Lagrangian.
        # log_alpha > 0  <=>  P(z > 0) > 0.5  <=>  anchor.
        # Distill loss pulls anchor heads to log_alpha >> 0;
        # L0 penalty pushes sparse heads to log_alpha << 0.
        # Natural two-polar convergence; log_alpha=0 is the exact split point.
        # gate_mat stores raw log_alpha; threshold at 0 gives a bool mask
        # that matches inference reuse_prefill.load_label (x.float() > 0.5).
        label = gate_mat > 0
    elif reg_mode == "hln":
        # HardLogistic(mu, s): gate_mat stores raw mu values.
        # mu > 0  <=>  sigmoid(mu) > 0.5  <=>  anchor.
        # Same threshold logic as hc (raw location param stored, threshold at 0).
        label = gate_mat > 0
    elif reg_mode == "hcl":
        # Hard Concrete + Lagrangian: gate_mat stores raw log_alpha values.
        # log_alpha > 0  <=>  sigmoid(log_alpha) > 0.5  <=>  P(z>0.5) > 0.5  <=>  anchor.
        # Lagrangian density = mean(sigmoid(alpha)) was driven to desired_density,
        # so at convergence fraction(alpha>0) ≈ desired_density (no export gap).
        label = gate_mat > 0
    else:
        label = gate_mat > 0.5
    label[0] = True  # layer 0 is forced all-anchor
    sparsity = 1.0 - label.float().mean().item()

    torch.save(label, os.path.join(output_dir, "label.pt"))
    save_full_attention_heads(
        full_attention_heads_to_list([g for g in gate_mat]),
        os.path.join(output_dir, "anchor_heads.tsv"),
    )
    meta = {
        "select_mode": SELECT_MODE,
        "budget": BUDGET,
        "block_size": BLOCK_SIZE,
        "segment_size": SEGMENT_SIZE,
        "sink_blocks": SINK_BLOCKS,
        "local_blocks": LOCAL_BLOCKS,
        "threshold": 0.5,
        "num_layers": int(gate_mat.shape[0]),
        "num_key_value_heads": int(gate_mat.shape[1]),
        "anchor_sparsity": sparsity,
        "layer0_forced_anchor": True,
        "reg_mode": reg_mode,
    }
    if reg_mode == "kuma" and desired_density is not None:
        meta["desired_density"] = float(desired_density)
    if reg_mode == "ddp":
        meta["target_sparsity"] = float(target_sparsity)
        meta["uniform_sparsity"] = bool(uniform_sparsity)
    if reg_mode == "stg":
        meta["target_sparsity"] = float(target_sparsity)
        meta["uniform_sparsity"] = bool(uniform_sparsity)
    if reg_mode == "hc":
        meta["target_sparsity"] = float(target_sparsity)
        meta["uniform_sparsity"] = bool(uniform_sparsity)
    if reg_mode == "hln":
        meta["target_sparsity"] = float(target_sparsity)
        meta["uniform_sparsity"] = bool(uniform_sparsity)
    if reg_mode == "hcl":
        meta["target_sparsity"] = float(target_sparsity)
        meta["uniform_sparsity"] = bool(uniform_sparsity)
    with open(os.path.join(output_dir, "reuse_v1_label_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return label, sparsity


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.model_name)

    config = AutoConfig.from_pretrained(args.config_name or args.model_name)
    if args.rope_theta is not None:
        config.rope_theta = args.rope_theta

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    enable_llama_reuse_v1_training(
        model,
        budget=BUDGET,
        block_size=BLOCK_SIZE,
        segment_size=SEGMENT_SIZE,
        sink_blocks=SINK_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
        initial_value=args.initial_value,
        causal=True,
        reg_mode=args.reg_mode,
        select_mode=args.select_mode,
        top_p=args.top_p,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
    )

    # Train only the gates; keep the shared holder reachable from the LlamaModel.
    holder = model._reuse_holder
    model = model.model
    model._reuse_holder = holder
    # Propagate stg_sigma to the holder so _kv_head_gate can read it.
    if args.reg_mode == "stg":
        holder.stg_sigma = getattr(args, "stg_sigma", 0.5)
    # hc mode: no extra holder state needed (beta/gamma/zeta are constants in hc_gate.py)

    for param in model.parameters():
        param.requires_grad = False
    num_gate = 0
    if args.reg_mode == "kuma":
        # kuma mode: the trainable params are the HardKuma shape params
        # (kuma_a / kuma_b); the deterministic anchor_heads gate is frozen. Layer
        # 0's shape params were frozen inside enable_* (all-anchor); leave them.
        for name, param in model.named_parameters():
            if name.endswith("kuma_a") or name.endswith("kuma_b"):
                if param.requires_grad is False and "layers.0." in name:
                    continue
                param.requires_grad = True
                num_gate += param.numel()
    else:
        for name, param in model.named_parameters():
            if name.endswith("anchor_heads"):
                # Layer 0's gate was frozen inside enable_*; leave it frozen.
                if param.requires_grad is False and "layers.0." in name:
                    continue
                param.requires_grad = True
                num_gate += param.numel()
                if args.reg_mode == "ddp":
                    # anchor_heads is the logit z_loga: init ~N(initial_value,
                    # z_loga_init_std) so heads start near-anchor but with a small
                    # spread that seeds the deterministic top-k ranking. Done here
                    # (before FSDP shards the params) and only for layers >=1
                    # (layer 0 stays frozen all-anchor at 1.0).
                    with torch.no_grad():
                        param.data.normal_(
                            mean=args.initial_value, std=args.z_loga_init_std)
                elif args.reg_mode == "stg":
                    # STG: mirror official FeatureSelector init (0.01*randn) to
                    # break head symmetry within each layer. Without this, all 8
                    # kv-heads start at the same mu=0 and receive identical gradients
                    # in the first few steps, slowing polarization. Done here
                    # (before FSDP shards) and only for layers >=1 (layer 0 frozen,
                    # already skipped above via requires_grad check).
                    with torch.no_grad():
                        param.data.normal_(mean=0.0, std=0.01)
                elif args.reg_mode == "hc":
                    # Hard Concrete (Louizos 2018 original): log_alpha init to
                    # args.initial_value (default 0.0 -> half of heads start near
                    # the alpha=0 boundary). Fixed L0 penalty (reg_weight * sum
                    # P(z!=0)) polarizes heads: distill loss pulls anchor heads to
                    # log_alpha >> 0, L0 penalty pushes sparse heads to log_alpha
                    # << 0. Export threshold log_alpha > 0 aligns with inference
                    # label > 0.5 in reuse_prefill.load_label. Small noise breaks
                    # per-layer head symmetry so rankings diverge quickly.
                    # Layer 0 is frozen (already skipped above).
                    with torch.no_grad():
                        param.data.normal_(mean=args.initial_value, std=0.01)
                elif args.reg_mode == "hln":
                    # HardLogistic(mu, s): mu init to args.initial_value (default 1.0
                    # -> sigmoid(1.0)=0.73 warm-start density; gradient=0.197, well
                    # away from saturation). Small noise breaks per-layer head
                    # symmetry (same as hc). Export threshold mu > 0 aligns with
                    # sigmoid(mu) > 0.5. Layer 0 frozen (already skipped above).
                    with torch.no_grad():
                        param.data.normal_(mean=args.initial_value, std=0.01)
                elif args.reg_mode == "hcl":
                    # HC + Lagrangian: log_alpha init to args.initial_value (same
                    # as hc). Warm-start near anchor so distill loss can quickly
                    # differentiate heads before the Lagrangian tightens. Export
                    # threshold log_alpha > 0 aligns with sigmoid(alpha) > 0.5.
                    with torch.no_grad():
                        param.data.normal_(mean=args.initial_value, std=0.01)

    setup()
    torch.cuda.set_device(local_rank)

    # --- Ulysses sequence parallelism setup ---
    # sp_size>1 shards one sequence across sp_size ranks; dp_size groups run
    # distinct samples. SP requires --two_pass (the fill/use split; AC only
    # kicks in there) and head counts divisible by sp_size. FSDP2 still shards
    # weights across the FULL flat world mesh, so the "* world_size" distill
    # recipe reduces gate grads correctly (SP contributions SUM via the
    # world-factor vs FSDP's average; distinct dp samples SUM into a global
    # mean-over-all-labels gradient). sp_size==1 leaves the old DP path intact.
    sp_size = getattr(args, "sp_size", 1)
    sp_group, sp_size, sp_rank, dp_size, dp_rank = _make_sp_group(
        sp_size, world_size, rank)
    if sp_size > 1:
        assert getattr(args, "two_pass", False), (
            "--sp_size > 1 requires --two_pass (SP is only wired for the "
            "two-pass fill/use forward).")
        enable_sequence_parallel(model, sp_group)
        if rank == 0:
            print(f"Ulysses SP enabled: sp_size={sp_size} dp_size={dp_size} "
                  f"(world={world_size}). Each SP group splits one sequence; "
                  f"FSDP2 shards weights over the full world mesh.")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    # NOTE: activation checkpointing is only safe in --two_pass mode. In that
    # mode Pass 2 ("use") only READS frozen holder.sels (no writes / no reset),
    # so recomputing a layer's forward during backward is a pure function and
    # reproduces identical results. In single-pass mode it would corrupt the
    # cross-layer top-k threading in the shared holder, so we leave it off.
    mesh = DeviceMesh(device_type="cuda", mesh=list(range(world_size)))

    # Optional grouped activation checkpointing: repack layers into groups BEFORE
    # FSDP so FSDP shards the (unchanged) inner LlamaDecoderLayers and sees the
    # final module tree. group_size>1 only makes sense under --two_pass (AC).
    ac_group_size = getattr(args, "ac_group_size", 1)
    if getattr(args, "two_pass", False) and ac_group_size > 1:
        _group_decoder_layers(model, ac_group_size)
        if rank == 0:
            print(f"Grouped decoder layers into segments of {ac_group_size} "
                  f"for activation checkpointing.")

    apply_fsdp(model, mesh, mp_policy, modules_to_shard={LlamaDecoderLayer})

    if getattr(args, "two_pass", False) and not getattr(args, "no_ac", False):
        from functools import partial
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        non_reentrant = partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        if ac_group_size > 1:
            ac_cls = _GroupedDecoderLayers
            ac_desc = f"grouped segments of {ac_group_size} LlamaDecoderLayers"
        else:
            ac_cls = LlamaDecoderLayer
            ac_desc = "LlamaDecoderLayer"
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant,
            check_fn=lambda m: isinstance(m, ac_cls),
        )
        if rank == 0:
            print(f"Two-pass mode: activation checkpointing enabled on "
                  f"{ac_desc} (non-reentrant).")
    elif getattr(args, "two_pass", False) and getattr(args, "no_ac", False):
        if rank == 0:
            print("Two-pass mode: activation checkpointing DISABLED (--no_ac). "
                  "Ensure per-rank activation memory fits (sp_size>1 recommended).")

    if rank == 0:
        print(f"Trainable gate params: {num_gate} across "
              f"{config.num_hidden_layers} layers "
              f"({config.num_key_value_heads} kv-heads each)")

    haystack_dataset = get_dataset(args.dataset_name, split="train")
    if args.dataset_format != "multiple_passkey":
        raise ValueError(f"Invalid dataset format: {args.dataset_format}")
    train_dataset = MultiplePasskeyRetrievalDataset(
        haystack_dataset,
        tokenizer,
        max_length=args.max_length,
        min_depth_ratio=args.min_needle_depth_ratio,
        max_depth_ratio=args.max_needle_depth_ratio,
        context_length_min=args.context_length_min,
        context_length_max=args.context_length_max,
        context_lengths_num_intervals=args.context_lengths_num_intervals,
        depth_ratio_num_intervals=args.depth_ratio_num_intervals,
        num_passkeys=args.num_passkeys,
    )
    # Data parallel over dp groups: every rank in an SP group must load the SAME
    # sample (they split one sequence), while distinct dp groups see distinct
    # samples. sp_size==1 -> dp_size==world_size, dp_rank==rank (old behavior).
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True
    )
    train_dataloader = get_supervised_dataloader(
        train_dataset, tokenizer, args.batch_size, sampler=sampler
    )

    # hln mode: create hl_s_raw BEFORE optimizer so it is included in param_groups.
    # s = softplus(s_raw) + HL_S_MIN; s_raw is a plain scalar (not FSDP-sharded).
    if args.reg_mode == "hln":
        import math as _math
        _hl_s_init = max(float(getattr(args, "hln_sigma", 0.5)), HL_S_MIN)
        _sp_target = _hl_s_init - HL_S_MIN
        _s_raw_init = _math.log(_math.expm1(max(_sp_target, 1e-6)))
        hl_s_raw = nn.Parameter(
            torch.tensor(_s_raw_init, dtype=torch.float32,
                         device=f"cuda:{local_rank}")
        )
        model._reuse_holder.hl_s_raw = hl_s_raw
    else:
        hl_s_raw = None

    # --- Optimizer. gates (+ kuma a/b) train under the primal lr; --reg_mode
    # ddp additionally learns the Lagrange multipliers by DUAL ASCENT (separate
    # maximize=True groups). The lambdas are plain (non-sharded) params created
    # identically on every rank; their gradient (s - target) is rank-identical
    # (s comes from the replicated full_tensor score), so they stay in sync with
    # no all-reduce. Mixing DTensor gate params and plain lambda params is safe:
    # AdamW's foreach path buckets tensors per param-group and per type. ---
    lambdas = None
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": args.lr}
    ]
    if args.reg_mode == "hln" and hl_s_raw is not None:
        # Add hl_s_raw to optimizer with same lr as gates. s trains slowly
        # because density is s-free; its gradient comes only from LM/distill
        # loss through z. A separate (potentially smaller) lr can be set here.
        param_groups.append({"params": [hl_s_raw], "lr": args.lr})
    if args.reg_mode == "ddp":
        dev = f"cuda:{local_rank}"
        # Init lambdas to a small positive value so the Lagrangian constraint
        # exerts sparsity pressure from step 1 (cold-start). With zeros the
        # reg_loss is 0 initially and the gate drifts toward all-anchor before
        # the dual-ascent can catch up. Value 1.0 mirrors kuma's lamda_init_value.
        _lambda_init = getattr(args, "lambda_init_value", 1.0)
        lambdas = nn.ParameterDict({
            "lambda_1": nn.Parameter(torch.full((1,), _lambda_init, device=dev)),
            "lambda_2": nn.Parameter(torch.full((1,), _lambda_init, device=dev)),
        })
        param_groups.append(
            {"params": [lambdas["lambda_1"]], "lr": args.lambda_1_lr,
             "maximize": True})
        param_groups.append(
            {"params": [lambdas["lambda_2"]], "lr": args.lambda_2_lr,
             "maximize": True})
        if not args.no_binary_loss:
            lambdas["lambda_3"] = nn.Parameter(torch.full((1,), _lambda_init, device=dev))
            l3_lr = (args.lambda_3_lr if args.lambda_3_lr is not None
                     else args.lambda_1_lr)
            param_groups.append(
                {"params": [lambdas["lambda_3"]], "lr": l3_lr,
                 "maximize": True})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0)
    warmup = max(1, args.num_steps // 5)

    def _warmup_lambda(step):
        return min(
            1,
            max((step + 1) / warmup, 0.1),
            max((args.num_steps - step) / warmup, 0.1),
        )

    if args.reg_mode == "ddp":
        # Only the gate group follows the warmup/decay schedule; the dual-ascent
        # lambda groups keep a constant lr (one lr_lambda per param group).
        lr_lambdas = ([_warmup_lambda]
                      + [(lambda step: 1.0)] * (len(param_groups) - 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambdas)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_warmup_lambda)

    if rank == 0 and args.output_dir is not None:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f)

    if rank == 0 and not args.disable_wandb:
        wandb.init(project="reuse_v1", config=vars(args))
        if args.exp_name is not None:
            wandb.run.name = args.exp_name

    resume_step = -1
    resume_path = (
        os.path.join(args.output_dir,
                     f"optimizer_scheduler_state_latest-rank={rank}.pt")
        if args.output_dir is not None else None
    )
    if args.resume and resume_path is not None and os.path.exists(resume_path):
        state = torch.load(resume_path)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        resume_step = state["global_step"]
        if rank == 0:
            print(f"Resuming from step {resume_step}")

    train(args, model, rank, world_size, train_dataloader, optimizer,
          scheduler, resume_step, tokenizer.pad_token_id, lambdas=lambdas)

    gates = _stack_gates(model)
    if args.reg_mode == "kuma":
        # Export decision statistic is the HardKuma mean (mean > 0.5 -> anchor),
        # not the frozen deterministic gate.
        a_full, b_full = _stack_kuma(model)
        gates = [
            hardkuma_mean(a.detach().cpu().clamp(1e-6, 100.0).float(),
                          b.detach().cpu().clamp(1e-6, 100.0).float())
            for a, b in zip(a_full, b_full)
        ]
    if rank == 0 and args.output_dir is not None:
        label, sparsity = _export_label(
            gates, args.output_dir, reg_mode=args.reg_mode,
            desired_density=getattr(args, "desired_density", None),
            target_sparsity=getattr(args, "target_sparsity", None),
            uniform_sparsity=getattr(args, "uniform_sparsity", False))
        print(f"Training finished. label shape={tuple(label.shape)} "
              f"anchor_sparsity={sparsity:.3f} -> {args.output_dir}/label.pt")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)

