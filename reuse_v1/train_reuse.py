"""Train reuse_v1 anchor/sparse head labels, DuoAttention-style.

Learns a per-(layer, kv-head) soft gate ``anchor_heads`` in [0,1] via two-way
distillation (teacher = dense block attention, student = gate-blended
dense/sparse) + L1 sparsity. At convergence, ``gate > 0.5`` -> anchor head;
otherwise sparse. Layer 0 is forced all-anchor in every ``--reg_mode`` (gate
frozen at anchor, excluded from the sparsity penalty), which guarantees each
kv-head has a preceding anchor selection to reuse.

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
    map_llama_anchor_heads,
)
# HardKuma density (Lagrangian target) + mean (export statistic) for --reg_mode kuma.
from kuma_gate import hardkuma_density, hardkuma_mean, NOISE_EPS
# Hard Concrete gate (--reg_mode hc): L0 penalty readout + deterministic export mask.
from hc_gate import (
    hc_p_nonzero,
    deterministic_head_mask,
    NOISE_EPS as HC_NOISE_EPS,
)

import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ---- LOCKED reuse_v1 hyperparameters (MUST match inference) ----
# The block-SELECTION config (select_mode / top_p / min_blocks / max_blocks) is
# NOT locked here: it comes from args and is recorded in the exported label
# metadata by _export_label.
BUDGET = 32
BLOCK_SIZE = 128
SEGMENT_SIZE = 2048
SINK_BLOCKS = 1
LOCAL_BLOCKS = 2


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
          resume_step, pad_token_id):
    model.train()
    if rank == 0:
        pbar = tqdm(range(args.num_steps))
    local_rank = int(os.environ["LOCAL_RANK"])

    global_step = 0
    local_step = 0

    reg_mode = getattr(model._reuse_holder, "reg_mode", "l1")
    # --- Lagrangian density-constraint state (--reg_mode kuma) ---
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
    elif reg_mode == "hc":
        # Hard Concrete: original Louizos 2018 formulation.
        # reg_loss = reg_weight * sum P(z_j != 0) = reg_weight * sum sigmoid(alpha+1.599)
        # Fixed coefficient reg_weight (from --reg_weight, same as l1 mode).
        # No Lagrangian. Sparsity at convergence is controlled by reg_weight:
        #   larger reg_weight → more heads pushed to alpha<<0 → more sparse.
        # Export: global top-k on log_alpha (monotone with P(z>0)) hitting
        # exactly --target_sparsity. The two-polar convergence (anchor heads
        # alpha>>0, sparse heads alpha<<0) is driven by the interplay of
        # distill_loss pulling important heads up and the L0 penalty pushing all
        # heads down.
        _gates0 = get_llama_anchor_heads(model)
        num_layers_hc = len(_gates0)
        hkv_hc = _gates0[0].shape[0]

    while True:
        if global_step >= args.num_steps:
            break
        for batch in train_dataloader:
            if global_step <= resume_step:
                global_step += 1
                if rank == 0:
                    pbar.update(1)
                continue

            # Per-step gate clamp. l1/kuma: the anchor_heads gate is a [0,1]
            # blend weight, so pin it to [0,1].
            if reg_mode == "hc":
                # log_alpha unconstrained in theory; clamp to prevent overflow
                # in sigmoid (sigmoid(±10) ≈ 1/0 to machine precision).
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
            elif reg_mode == "hc":
                # Freeze uniform noise for the WHOLE step (same pattern as kuma).
                gen = torch.Generator(device=f"cuda:{local_rank}")
                gen.manual_seed(args.seed + global_step)
                u = torch.rand(
                    num_layers_hc, hkv_hc,
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
            elif reg_mode == "hc":
                la_full = _stack_gates(model)
                la_full = [g.to(teacher_h.device) for g in la_full]
                # Exclude layer 0 (frozen all-anchor) from the L0 sparsity penalty.
                la_cat = torch.cat(la_full[1:]).float()
                l0_reg = hc_p_nonzero(la_cat).sum()
                reg_loss = args.reg_weight * l0_reg
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

            do_save = (args.output_dir is not None
                       and global_step % args.save_steps == 0)
            gates_list = None
            gate_state = None
            if rank == 0:
                # hc: snapshot log_alpha before full_attention_heads_to_list
                # converts gate tensors to plain Python lists in-place.
                if reg_mode == "hc":
                    # Layer 0 is frozen all-anchor; report density over the
                    # trainable layers only, matching the L0 penalty.
                    _la_snap = torch.cat([g.detach().float() for g in gates[1:]])
                    _p_nz = hc_p_nonzero(_la_snap)
                    # True Hard Concrete L0 density: E[fraction of heads with
                    # z != 0] = mean sigmoid(log_alpha + 1.5986). NOT
                    # (log_alpha > 0), whose threshold corresponds to
                    # P(z != 0) = 0.83 and badly under-reports the density.
                    _l0_density = _p_nz.mean().item()
                    # E[fraction of heads whose gate fires the anchor-cache
                    # write this step]. The write rule is z[h] > 0.5, which
                    # fires with probability sigmoid(log_alpha), so this is the
                    # expected write COUNT / num_heads -- the zero-variance
                    # version of (z > 0.5).float().mean(). Compare against the
                    # deploy anchor fraction over the same layers 1..L-1:
                    #   w_target = (L*H*(1 - target_sparsity) - H) / ((L-1)*H)
                    _write_frac = torch.sigmoid(_la_snap).mean().item()

                if do_save:
                    # Build the resumable gate state from the already-gathered
                    # full tensors, BEFORE full_attention_heads_to_list replaces
                    # them with Python lists.
                    if reg_mode == "kuma":
                        gate_state = {
                            "reg_mode": "kuma",
                            "kuma_a": [a.detach().float().cpu() for a in a_full],
                            "kuma_b": [b.detach().float().cpu() for b in b_full],
                        }
                    else:
                        gate_state = {
                            "reg_mode": reg_mode,
                            "anchor_heads": [g.detach().float().cpu()
                                             for g in gates],
                        }

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
                    if reg_mode == "hc":
                        # E[fraction of heads with z != 0] (true HC L0 density).
                        log_dict["l0_density"] = _l0_density
                        # Expected fraction of heads writing the anchor cache
                        # this step (= mean sigmoid(log_alpha)).
                        log_dict["write_frac"] = _write_frac
                        log_dict["l0_reg"] = float(reg_loss.item())
                    wandb.log(log_dict, step=global_step)
                    plt.close(fig)

                if reg_mode == "kuma":
                    extra = (f"|dens={float(density.item()):.3f}"
                             f"|lam={float(lambda0.item()):.2e}")
                elif reg_mode == "hc":
                    extra = (f"|l0={_l0_density:.3f}|w={_write_frac:.3f}"
                             f"|rw={args.reg_weight:.4f}")
                else:
                    extra = ""
                pbar.set_description(
                    f"Len={seq_len}|N={int(global_num_labels)}"
                    f"|D={distill_loss.item():.3f}|R={reg_loss.item():.3f}"
                    f"{extra}|LR={optimizer.param_groups[0]['lr']:.2e}"
                )
                pbar.update(1)

            # Outside the rank-0 block: EVERY rank must write its own optimizer
            # shard, or a later --resume desyncs the ranks.
            if do_save:
                _save_checkpoint(args, gates_list, gate_state, optimizer,
                                 scheduler, global_step, rank)

            if global_step >= args.num_steps:
                break

    if rank == 0:
        pbar.close()


def _save_checkpoint(args, gates_list, gate_state, optimizer, scheduler,
                     global_step, rank):
    """Write a resumable checkpoint.

    EVERY rank writes its own ``optimizer_scheduler_state_latest-rank={rank}.pt``:
    under FSDP2 the gate params are DTensors, so each rank owns a distinct shard
    of the Adam moments. If only rank 0 wrote, a later ``--resume`` would leave
    ranks 1..N-1 with ``resume_step = -1`` and they would desync from rank 0's
    dataloader skip loop.

    Rank 0 additionally writes ``gate_state_latest.pt`` (the full, unsharded
    trainable gate values) and the human-readable tsvs. The gate values are NOT
    in the optimizer state dict, so without this file a resume would silently
    restart from the random init.
    """
    if rank == 0:
        save_full_attention_heads(
            gates_list,
            os.path.join(args.output_dir, f"anchor_heads_step={global_step}.tsv"),
        )
        latest = os.path.join(args.output_dir, "anchor_heads_latest.tsv")
        save_full_attention_heads(gates_list, latest)
        gate_state = dict(gate_state)
        gate_state["global_step"] = global_step
        torch.save(gate_state,
                   os.path.join(args.output_dir, "gate_state_latest.pt"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        },
        os.path.join(args.output_dir,
                     f"optimizer_scheduler_state_latest-rank={rank}.pt"),
    )


def _restore_gate_state(model, gate_state, reg_mode):
    """Copy saved full gate values back into the (still unsharded) gate params.

    MUST be called BEFORE ``apply_fsdp``: at that point ``anchor_heads`` /
    ``kuma_a`` / ``kuma_b`` are plain tensors, so a straight ``copy_`` works and
    no DTensor redistribution is needed. Must also run AFTER the hc
    ``normal_(initial_value, 0.01)`` re-init, which would otherwise clobber the
    restored values.
    """
    saved_mode = gate_state.get("reg_mode", "l1")
    if saved_mode != reg_mode:
        raise ValueError(
            f"gate_state_latest.pt was written with --reg_mode {saved_mode!r} "
            f"but this run uses {reg_mode!r}; refusing to resume.")
    if reg_mode == "kuma":
        a_list, b_list = get_llama_kuma_params(model)
        pairs = list(zip(a_list, gate_state["kuma_a"])) + \
                list(zip(b_list, gate_state["kuma_b"]))
    else:
        pairs = list(zip(get_llama_anchor_heads(model),
                         gate_state["anchor_heads"]))
    with torch.no_grad():
        for param, saved in pairs:
            param.copy_(torch.as_tensor(saved).to(param.device, param.dtype))
    return gate_state.get("global_step", -1)


def _export_label(gates, output_dir, reg_mode="l1", desired_density=None,
                  target_sparsity=None, uniform_sparsity=False,
                  select_mode="topk", top_p=None, min_blocks=None,
                  max_blocks=None):
    """Write the reuse_v1 label: (num_layers, Hkv) bool tensor.

    Saves ``label.pt`` (bool tensor consumed by reuse_prefill.load_label),
    ``anchor_heads.tsv`` (raw float gates for inspection) and ``config.json``
    with the locked hyperparameters. ``gates`` are the per-layer anchor decision
    statistics: the deterministic gate (l1 mode) or the HardKuma mean (kuma
    mode), thresholded at ``> 0.5``. For ``hc`` mode: gates are log_alpha values
    and a deterministic top-k head mask hits the EXACT ``target_sparsity``.
    Layer 0 is forced all-anchor in every mode, so the ``target_sparsity`` zeros
    are drawn from layers 1..L-1 only.

    ``select_mode`` / ``top_p`` / ``min_blocks`` / ``max_blocks`` are the ACTUAL
    block-selection settings used for training (from args, not module constants);
    they are written into ``reuse_v1_label_config.json`` because inference MUST
    reproduce them or the exported head labels do not transfer.
    """
    gate_mat = torch.stack([g.detach().float().cpu() for g in gates], dim=0)
    if reg_mode == "hc":
        hard = deterministic_head_mask(
            gate_mat, target_sparsity, uniform_sparsity=uniform_sparsity,
            layer0_forced=True)
        label = hard > 0.5
    else:
        label = gate_mat > 0.5
    label[0] = True  # all modes: layer 0 forced all-anchor
    sparsity = 1.0 - label.float().mean().item()

    torch.save(label, os.path.join(output_dir, "label.pt"))
    save_full_attention_heads(
        full_attention_heads_to_list([g for g in gate_mat]),
        os.path.join(output_dir, "anchor_heads.tsv"),
    )
    meta = {
        "select_mode": select_mode,
        "top_p": float(top_p) if top_p is not None else None,
        "min_blocks": int(min_blocks) if min_blocks is not None else None,
        "max_blocks": int(max_blocks) if max_blocks is not None else None,
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
    if reg_mode == "hc":
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
                # All modes: layer 0 was frozen all-anchor inside enable_* and
                # must stay frozen (and keep its forced value, so skip the hc
                # re-init below too).
                if "layers.0." in name:
                    continue
                param.requires_grad = True
                num_gate += param.numel()
                if args.reg_mode == "hc":
                    # Hard Concrete: log_alpha init to args.initial_value with
                    # small noise to break per-layer head symmetry.
                    with torch.no_grad():
                        param.data.normal_(mean=args.initial_value, std=0.01)

    # --- Resume the learned gate values (all ranks, BEFORE FSDP shards them) ---
    # The gate values are not part of the optimizer state dict, so this file is
    # the only thing that carries them across a restart. Doing it here (params
    # still plain tensors, after the hc re-init above) keeps it a simple copy_.
    gate_resume_step = -1
    if args.resume and args.output_dir is not None:
        gate_path = os.path.join(args.output_dir, "gate_state_latest.pt")
        if os.path.exists(gate_path):
            gate_resume_step = _restore_gate_state(
                model, torch.load(gate_path, map_location="cpu"), args.reg_mode)
            if rank == 0:
                print(f"Restored gate values from {gate_path} "
                      f"(saved at step {gate_resume_step}).")
        elif rank == 0:
            print(f"WARNING: --resume set but {gate_path} not found; the gates "
                  f"will restart from their initial values.")

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

    # --- Optimizer: gates (+ kuma a/b) train under the primal lr. ---
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": args.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0)
    warmup = max(1, args.num_steps // 5)

    def _warmup_lambda(step):
        return min(
            1,
            max((step + 1) / warmup, 0.1),
            max((args.num_steps - step) / warmup, 0.1),
        )

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
    loaded_opt_state = False
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
        loaded_opt_state = True
    # All ranks MUST agree on resume_step: train() skips batches until
    # global_step > resume_step, and a mismatch desyncs the collectives. Rank 0's
    # value wins (a checkpoint written before every rank saved leaves ranks
    # 1..N-1 without a file, which used to deadlock on the first collective).
    _rs = torch.tensor([resume_step], device=f"cuda:{local_rank}")
    dist.broadcast(_rs, src=0)
    resume_step = int(_rs.item())
    if args.resume and resume_step >= 0 and not loaded_opt_state:
        # Fast-forward the LR schedule by hand: the skip loop in train() does not
        # call scheduler.step(), so without this the rank would run at the
        # warmup LR while rank 0 runs at the resumed LR -- different updates on
        # different shards of the same gate.
        for _ in range(resume_step + 1):
            scheduler.step()
        print(f"WARNING: [rank {rank}] {resume_path} missing; resuming at step "
              f"{resume_step} WITHOUT this rank's Adam moments (LR schedule "
              f"fast-forwarded).")
    if rank == 0 and resume_step >= 0:
        print(f"Resuming from step {resume_step}")

    train(args, model, rank, world_size, train_dataloader, optimizer,
          scheduler, resume_step, tokenizer.pad_token_id)

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
            uniform_sparsity=getattr(args, "uniform_sparsity", False),
            select_mode=args.select_mode, top_p=args.top_p,
            min_blocks=args.min_blocks, max_blocks=args.max_blocks)
        print(f"Training finished. label shape={tuple(label.shape)} "
              f"anchor_sparsity={sparsity:.3f} -> {args.output_dir}/label.pt")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)

