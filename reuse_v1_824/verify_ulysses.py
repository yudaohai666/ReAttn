"""Verify Ulysses sequence parallelism for reuse_v1 training.

Launch under torchrun (N ranks = the SP group; no FSDP, each rank holds the full
model). Two tiers:

--tier eq   [--eq-len 4096]
    Correctness. All ranks run the SAME (broadcast) input. Each rank runs the SP
    two-pass forward on its sequence shard, computes the SP loss
    (local_mse_sum / global_num_labels  +  reg_weight * L1 / sp_size), backward,
    then all-reduce-SUMs the per-(layer,kv-head) gate gradients across the SP
    group. Rank 0 THEN runs the reference: the identical model with SP OFF, on
    the FULL sequence, standard single-GPU two-pass loss + backward. The SP gate
    gradients must match the single-GPU gate gradients to bf16 backprop
    tolerance (global cosine ~1, relative-L2 small). Also checks the gathered
    student hidden states match. This validates the whole SP machinery:
    all-to-all layout, global top-k selection per head shard, cross-rank
    gradient flow through the collective, and the gate-grad reduction recipe.

--tier mem  [--len 131072]
    Feasibility. Run a few SP two-pass steps at a long length on N ranks; report
    peak GPU memory and step time. This is the path that replaces CPU offload.

Usage (project venv, H800):
    torchrun --nproc_per_node 8 reuse_v1/verify_ulysses.py --tier eq
    torchrun --nproc_per_node 8 reuse_v1/verify_ulysses.py --tier mem --len 131072
    # Qwen (4 kv-heads) -> sp=4:
    torchrun --nproc_per_node 4 reuse_v1/verify_ulysses.py --tier eq \
        --model-path <qwen>
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor import DeviceMesh

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATCH_DIR = os.path.join(_REPO_ROOT, "reuse_v1", "patch")
_RV1_DIR = os.path.join(_REPO_ROOT, "reuse_v1")
for _p in (_REPO_ROOT, _PATCH_DIR, _RV1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from reuse_train_llama import (
    enable_llama_reuse_v1_training,
    enable_sequence_parallel,
    get_llama_anchor_heads,
    map_llama_anchor_heads,
)
from sp_ulysses import SPContext

BUDGET = 32
BLOCK_SIZE = 128
SEGMENT_SIZE = 2048
SINK_BLOCKS = 1
LOCAL_BLOCKS = 2

_DEFAULT_MODEL = os.environ.get(
    "LLAMA_REUSE_V1_MODEL_PATH",
    "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
)


def _build_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(device)
    enable_llama_reuse_v1_training(
        model, budget=BUDGET, block_size=BLOCK_SIZE, segment_size=SEGMENT_SIZE,
        sink_blocks=SINK_BLOCKS, local_blocks=LOCAL_BLOCKS,
        initial_value=1.0, causal=True,
    )
    holder = model._reuse_holder
    inner = model.model  # LlamaModel
    inner._reuse_holder = holder
    for p in inner.parameters():
        p.requires_grad = False
    for name, p in inner.named_parameters():
        if name.endswith("anchor_heads"):
            if "layers.0." in name:
                continue
            p.requires_grad = True
    # trainable (non-layer-0) gates -> 0.5 so the sparse branch actually
    # contributes (else student==dense==teacher and grads are trivially 0).
    with torch.no_grad():
        for L, g in enumerate(get_llama_anchor_heads(inner)):
            if L == 0:
                continue
            g.fill_(0.5)
    return inner, holder


def _clamp_gates(model):
    @torch.no_grad()
    def clamp_(x):
        x.clamp_(0, 1)
    map_llama_anchor_heads(model, clamp_)


def _zero_gate_grads(model):
    for g in get_llama_anchor_heads(model):
        g.grad = None


def _gate_grads(model):
    return [
        (g.grad.detach().float().cpu().clone() if g.grad is not None else None)
        for g in get_llama_anchor_heads(model)
    ]


def _positions(lo, hi, device):
    return torch.arange(lo, hi, device=device).unsqueeze(0)


def _apply_ac(model):
    """Non-reentrant activation checkpoint on every LlamaDecoderLayer so the
    grad-retaining use-pass keeps only layer-boundary activations (recompute on
    backward). Orthogonal to SP/FSDP; mirrors verify_two_pass / train_reuse."""
    from functools import partial
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
    )
    non_reentrant = partial(checkpoint_wrapper,
                            checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant,
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
    )


def _apply_fsdp(model, mesh, reshard=True):
    """FSDP2 (fully_shard) over the SAME mesh as SP: shards 8B bf16 weights /
    grads / optim across ranks (~16GB -> ~2GB/rank).
    Mirrors train_reuse.apply_fsdp / duo_attn.train.apply_fsdp.

    reshard=True  -> reshard_after_forward=True: params re-all-gathered EVERY
                     forward (3 gathers across the 3 reuse passes).
    reshard=False -> params stay UNSHARDED (~16GB/rank) after the first gather;
                     the two no_grad fill passes + the use pass + backward all
                     reuse them -> 2x fewer all-gathers. Costs +14GB/rank
                     resident, cheap under the 66GB 128k headroom."""
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                              reduce_dtype=torch.bfloat16)
    cfg = {"mp_policy": mp, "mesh": mesh, "reshard_after_forward": reshard}
    for m in model.modules():
        if isinstance(m, LlamaDecoderLayer):
            fully_shard(m, **cfg)
    fully_shard(model, **cfg)


def _stack_gates_full(model):
    """Per-layer (Hkv,) gates gathered to full (unsharded) tensors (DTensor ->
    local) so the L1 reg is identical on every rank; FSDP averaging of identical
    grads is a no-op, giving the exact single-GPU reg gradient."""
    gates = get_llama_anchor_heads(model)
    return [g.full_tensor() if hasattr(g, "full_tensor") else g for g in gates]


# PLACEHOLDER_RUNNERS


def run_sp_two_pass(model, holder, input_ids_full, label_mask_full, group,
                    reg_weight):
    """SP two-pass step on this rank's sequence shard. Returns (student_h_local
    detached, distill_loss scalar, summed gate grads list). Gate grads are
    all-reduce-SUMmed across the SP group so every rank holds the full grad."""
    sp = holder.sp
    P, r = sp.sp_size, sp.sp_rank
    device = input_ids_full.device
    seq_len = input_ids_full.shape[1]
    s_local = seq_len // P
    lo, hi = r * s_local, (r + 1) * s_local
    ids_local = input_ids_full[:, lo:hi].contiguous()
    mask_local = label_mask_full[:, lo:hi].contiguous()
    pos = _positions(lo, hi, device)

    _clamp_gates(model)
    _zero_gate_grads(model)

    holder.phase = "fill_teacher"
    with torch.no_grad():
        th = model(input_ids=ids_local, position_ids=pos)[0]
    holder.phase = "fill_student"
    with torch.no_grad():
        model(input_ids=ids_local, position_ids=pos)
    holder.phase = "use"
    sh = model(input_ids=ids_local, position_ids=pos)[0]

    # global label count across the SP group
    local_n = mask_local.sum()
    global_n = local_n.clone()
    dist.all_reduce(global_n, group=group)
    global_n = global_n.clamp_min(1)

    t = th[mask_local].float()
    s = sh[mask_local].float()
    local_mse_sum = (t - s).pow(2).mean(dim=-1).sum()  # connected even if empty
    distill_local = local_mse_sum / global_n

    gates = get_llama_anchor_heads(model)
    reg = torch.cat([g.float() for g in gates[1:]]).abs().sum()
    loss = distill_local + reg_weight * (reg / P)
    loss.backward()

    # assemble the full gate gradient: each rank produced grad only for its own
    # kv-head slice (distill) + full reg/P; SUM over the SP group -> exact
    # single-GPU gradient.
    for g in gates:
        if g.grad is not None:
            dist.all_reduce(g.grad, group=group)

    # report the true (global) distill loss for logging
    distill_global = local_mse_sum.detach().clone()
    dist.all_reduce(distill_global, group=group)
    distill_global = (distill_global / global_n).item()
    return sh.detach().float(), distill_global, _gate_grads(model)


def run_ref_two_pass(model, holder, input_ids_full, label_mask_full, reg_weight):
    """Single-GPU reference (SP OFF) on the FULL sequence. Standard two-pass."""
    holder.sp = SPContext(None)  # disable SP for the reference
    device = input_ids_full.device
    seq_len = input_ids_full.shape[1]
    pos = _positions(0, seq_len, device)

    _clamp_gates(model)
    _zero_gate_grads(model)

    holder.phase = "fill_teacher"
    with torch.no_grad():
        th = model(input_ids=input_ids_full, position_ids=pos)[0]
    holder.phase = "fill_student"
    with torch.no_grad():
        model(input_ids=input_ids_full, position_ids=pos)
    holder.phase = "use"
    sh = model(input_ids=input_ids_full, position_ids=pos)[0]

    n = label_mask_full.sum().clamp_min(1)
    t = th[label_mask_full].float()
    s = sh[label_mask_full].float()
    distill = (t - s).pow(2).mean(dim=-1).sum() / n
    gates = get_llama_anchor_heads(model)
    reg = torch.cat([g.float() for g in gates[1:]]).abs().sum()
    loss = distill + reg_weight * reg
    loss.backward()
    return sh.detach().float().cpu(), distill.item(), _gate_grads(model)


def _pad_to(seq_len, multiple):
    r = seq_len % multiple
    return seq_len if r == 0 else seq_len + (multiple - r)


def run_sp_two_pass_fsdp(model, holder, input_ids_full, label_mask_full, group,
                         reg_weight, world):
    """SP + FSDP2 two-pass step (mem-tier recipe, mirrors train_reuse.train).

    Sequence is SP-sharded across the mesh; weights are FSDP-sharded across the
    SAME mesh. Loss recipe compensates FSDP gradient *averaging*:
      distill = local_mse_sum * world / global_num_labels
      reg     = L1(full_tensor(gates[1:]))        (rank-identical -> avg no-op)
    backward() then lets FSDP reduce-scatter the gate grads -- NO manual
    all-reduce (that is the pure-SP eq-tier recipe, not this one). Returns the
    global (AVG-reduced) distill scalar for logging.
    """
    sp = holder.sp
    P, r = sp.sp_size, sp.sp_rank
    device = input_ids_full.device
    seq_len = input_ids_full.shape[1]
    s_local = seq_len // P
    lo, hi = r * s_local, (r + 1) * s_local
    ids_local = input_ids_full[:, lo:hi].contiguous()
    mask_local = label_mask_full[:, lo:hi].contiguous()
    pos = _positions(lo, hi, device)

    _clamp_gates(model)

    holder.phase = "fill_teacher"
    with torch.no_grad():
        th = model(input_ids=ids_local, position_ids=pos)[0]
    holder.phase = "fill_student"
    with torch.no_grad():
        model(input_ids=ids_local, position_ids=pos)
    holder.phase = "use"
    sh = model(input_ids=ids_local, position_ids=pos)[0]

    local_n = mask_local.sum()
    global_n = local_n.clone().detach()
    dist.all_reduce(global_n, group=group)
    global_n = global_n.clamp_min(1)

    # empty-mask stays graph-connected so label-free ranks still traverse the
    # collective backward in lockstep.
    t = th[mask_local].float()
    s = sh[mask_local].float()
    distill = (t - s).pow(2).mean(dim=-1).sum() * world / global_n

    gates = _stack_gates_full(model)
    gates = [g.to(s.device) for g in gates]
    reg = torch.cat([g.float() for g in gates[1:]]).abs().sum()
    loss = distill + reg_weight * reg
    loss.backward()

    distill_log = distill.detach().clone()
    dist.all_reduce(distill_log, op=dist.ReduceOp.AVG, group=group)
    return distill_log.item()


def _compare_grads(sp_grads, ref_grads):
    """Worst-case max|Δ|, global cosine, global relative-L2 over all
    non-None gate grads (skips layer-0 frozen None)."""
    worst_abs = 0.0
    dots = num_sp = num_ref = cross = 0.0
    sp_flat, ref_flat = [], []
    for gs, gr in zip(sp_grads, ref_grads):
        if gs is None or gr is None:
            continue
        d = (gs - gr).abs().max().item()
        worst_abs = max(worst_abs, d)
        sp_flat.append(gs.flatten())
        ref_flat.append(gr.flatten())
    if not sp_flat:
        return worst_abs, 1.0, 0.0
    a = torch.cat(sp_flat)
    b = torch.cat(ref_flat)
    cos = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)).item()
    rel_l2 = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    return worst_abs, cos, rel_l2


def tier_eq(args, rank, world, group, device):
    reg_weight = 1e-3
    eq_len = _pad_to(args.eq_len, max(world, BLOCK_SIZE))
    assert eq_len % world == 0 and eq_len % BLOCK_SIZE == 0
    if rank == 0:
        print(f"[eq] world={world} eq_len={eq_len} model={args.model_path}")

    model, holder = _build_model(args.model_path, device)

    # identical input on all ranks (broadcast from rank 0)
    vocab = min(model.config.vocab_size, 32000)
    input_ids = torch.randint(0, vocab, (1, eq_len), device=device)
    dist.broadcast(input_ids, src=0, group=group)
    label_mask = torch.zeros(1, eq_len, dtype=torch.bool, device=device)
    label_mask[:, -256:] = True

    enable_sequence_parallel(model, group)
    sh_local, distill_sp, sp_grads = run_sp_two_pass(
        model, holder, input_ids, label_mask, group, reg_weight)

    # gather local student-hidden shards -> full sequence (global order)
    P = world
    gathered = [torch.empty_like(sh_local) for _ in range(P)]
    dist.all_gather(gathered, sh_local.contiguous(), group=group)
    sh_sp_full = torch.cat(gathered, dim=1).cpu()  # (1, s_full, hidden)

    if rank != 0:
        return

    # free SP-side GPU activations/tensors before the full-seq reference
    del gathered, sh_local
    _zero_gate_grads(model)
    torch.cuda.empty_cache()

    # single-GPU reference (SP OFF) on the full sequence
    sh_ref, distill_ref, ref_grads = run_ref_two_pass(
        model, holder, input_ids, label_mask, reg_weight)

    h_abs = (sh_sp_full - sh_ref).abs().max().item()
    h_rel = ((sh_sp_full - sh_ref).norm() / sh_ref.norm().clamp_min(1e-12)).item()
    worst_abs, cos, rel_l2 = _compare_grads(sp_grads, ref_grads)

    print(f"[eq] distill  sp={distill_sp:.6f}  ref={distill_ref:.6f}  "
          f"|Δ|={abs(distill_sp - distill_ref):.3e}")
    print(f"[eq] student-hidden  max|Δ|={h_abs:.3e}  rel-L2={h_rel:.3e}")
    print(f"[eq] gate-grad  worst max|Δ|={worst_abs:.3e}  "
          f"cosine={cos:.8f}  rel-L2={rel_l2:.3e}")

    # bf16 backprop tolerance: gate grads are the definitive metric.
    ok = (cos > 0.9999) and (rel_l2 < 1e-2) and (h_rel < 1e-2)
    print(f"[eq] {'PASS' if ok else 'FAIL'}")


def tier_mem(args, rank, world, group, device):
    reg_weight = 1e-3
    seq_len = _pad_to(args.len, max(world, BLOCK_SIZE))
    assert seq_len % world == 0 and seq_len % BLOCK_SIZE == 0
    if rank == 0:
        print(f"[mem] world={world} seq_len={seq_len} model={args.model_path}")

    model, holder = _build_model(args.model_path, device)
    # Order (matches duo_attn / train_reuse): SP patch -> AC -> FSDP2, all on the
    # SAME flat mesh. requires_grad was set in _build_model (before fully_shard,
    # so FSDP tracks the trainable gates correctly).
    enable_sequence_parallel(model, group)
    if not args.no_ac:
        _apply_ac(model)
    mesh = DeviceMesh(device_type="cuda", mesh=list(range(world)))
    _apply_fsdp(model, mesh, reshard=(not args.no_reshard))
    if rank == 0:
        print(f"[mem] SP+FSDP2 on LlamaDecoderLayer (same mesh)  "
              f"AC={'off' if args.no_ac else 'on'}  "
              f"reshard_after_forward={not args.no_reshard}")

    vocab = min(model.config.vocab_size, 32000)
    torch.manual_seed(1234 + rank)
    input_ids = torch.randint(0, vocab, (1, seq_len), device=device)
    dist.broadcast(input_ids, src=0, group=group)
    label_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    label_mask[:, -256:] = True

    n_steps = args.steps
    for step in range(n_steps):
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier(group=group)
        t0 = time.time()
        distill = run_sp_two_pass_fsdp(
            model, holder, input_ids, label_mask, group, reg_weight, world)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        peak_t = torch.tensor([peak], device=device)
        dist.all_reduce(peak_t, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            print(f"[mem] step {step}  time={dt:.2f}s  "
                  f"peak(max over ranks)={peak_t.item():.1f}GB  "
                  f"distill={distill:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["eq", "mem"], default="eq")
    ap.add_argument("--eq-len", type=int, default=4096)
    ap.add_argument("--len", type=int, default=131072)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--model-path", default=_DEFAULT_MODEL)
    ap.add_argument("--no-ac", action="store_true",
                    help="mem tier: disable activation checkpointing on the use "
                    "pass (keeps all activations, saves the recompute pass; "
                    "trades GPU memory for speed).")
    ap.add_argument("--no-reshard", action="store_true",
                    help="mem tier: reshard_after_forward=False so FSDP keeps "
                    "weights unsharded across the 3 reuse passes (2x fewer "
                    "all-gathers; +14GB/rank resident).")
    args = ap.parse_args()

    # long timeout: the first (warmup) step at long context can spend minutes
    # in Triton autotune between collectives; don't let the NCCL watchdog kill it.
    dist.init_process_group(backend="nccl",
                            timeout=timedelta(seconds=3600))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group = dist.group.WORLD

    try:
        if args.tier == "eq":
            tier_eq(args, rank, world, group, device)
        else:
            tier_mem(args, rank, world, group, device)
    finally:
        dist.barrier(group=group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

