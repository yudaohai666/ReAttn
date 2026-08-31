"""Deterministic differentiable gate for reuse_v1 '--reg_mode ddp' training.

A pared-down port of LycheeDecode's ``train_ddp_v3.py`` gate machinery, exposing
only what the reuse_v1 two-way trainer needs. Mirrors ``kuma_gate.py`` in shape
(pure functions, no module state).

Polarity matches the deterministic reuse_v1 gate and the kuma gate: z=1 -> anchor
(dense), z=0 -> sparse (cheap). ``score`` is the annealed soft-saturation readout
used by the Lagrangian; ``expected_sparsity = 1 - score.mean()``.

The DDP gate reuses reuse_v1's per-(layer,kv-head) ``anchor_heads`` Param as the
learnable logit ``z_loga`` (init ~N(initial_value, z_loga_init_std)). The training
forward gate is ``ste_clamp(z_loga, 0, 1)`` (straight-through), layer 0 forced 1.
At export, ``deterministic_head_mask`` zeroes the lowest-scoring heads to hit the
exact target sparsity (layer 0 always anchor).

Excluded from this port (per design): ``build_train_gate`` (stochastic path) and
``per_layer_distill_loss`` (depth-debiased energy-normalized MSE) -- reuse_v1 keeps
its own label-mask distill loss.
"""

import torch

# HardConcrete/DDP stretch support (Louizos et al.); matches train_ddp_v3.
LIMIT_A, LIMIT_B = -0.1, 1.1


def ste_relu(x):
    """Straight-through relu: forward = relu(x), backward = identity."""
    x_hard = torch.relu(x)
    return x + (x_hard - x).detach()


def ste_clamp(x, min_val=0.0, max_val=1.0):
    """Straight-through clamp: forward = clamp, backward = identity."""
    x_hard = torch.clamp(x, min_val, max_val)
    return x + (x_hard - x).detach()


def soft_saturation_score(z_loga, mean):
    """Annealed soft-saturation readout in [0, 1], layer 0 forced all-anchor.

    Sharpens toward a hard 0/1 gate as ``mean`` anneals 0.5 -> 0.1. Used for the
    Lagrangian density constraint (``expected_sparsity = 1 - score.mean()``).
    """
    scale = 2.4 / mean
    z = ste_relu(z_loga)
    z = torch.sigmoid(scale * (z - mean))
    z = z * (LIMIT_B - LIMIT_A) + LIMIT_A
    z = ste_clamp(z, 0.0, 1.0)
    H = z_loga.shape[1]
    ones0 = torch.ones(1, H, device=z_loga.device, dtype=z_loga.dtype)
    z = torch.cat([ones0, z[1:]], dim=0)
    return z


def anneal_mean(
    progress,
    schedule="sqrt",
    mean_min=0.1,
    mean_max=0.5,
    warmup_ratio=0.0,
):
    """Anneal the soft-saturation ``mean`` from ``mean_max`` down to ``mean_min``."""
    if warmup_ratio > 0.0:
        if progress <= warmup_ratio:
            return mean_max
        progress = (progress - warmup_ratio) / max(1e-8, 1.0 - warmup_ratio)
    if schedule == "sqrt":
        frac = progress ** 0.5
    elif schedule == "linear":
        frac = progress
    elif schedule == "quad":
        frac = progress ** 2
    else:
        raise ValueError(f"unknown anneal schedule: {schedule}")
    return mean_max - (mean_max - mean_min) * frac


def deterministic_head_mask(z_loga, target_sparsity, uniform_sparsity=False):
    """Deploy readout: zero the lowest-scoring heads to hit exact target sparsity.

    Returns a hard {0,1} mask of shape z_loga.shape; 1 -> anchor (dense), 0 ->
    sparse. Layer 0 is always all-anchor. With ``uniform_sparsity`` each layer
    zeros the same per-layer count; otherwise a single global top-k over all
    (layer>=1, head) slots.
    """
    L, H = z_loga.shape
    soft = ste_relu(z_loga).detach().clone()
    hard = torch.ones_like(soft)
    if uniform_sparsity:
        k = round(target_sparsity * H)
        for layer in range(1, L):
            row = soft[layer].clone()
            marks = torch.ones(H, device=soft.device)
            if k > 0:
                _, idx = torch.topk(row, k=min(k, H), largest=False)
                marks[idx] = 0.0
            hard[layer] = marks
    else:
        soft[0] = float("inf")
        num_zeros = round(target_sparsity * L * H)
        flat = soft.reshape(-1).clone()
        if num_zeros > 0:
            _, idx = torch.topk(flat, k=min(num_zeros, flat.numel()), largest=False)
            hard.reshape(-1)[idx] = 0.0
    hard[0, :] = 1.0
    return hard
