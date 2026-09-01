"""Hard Concrete gate for reuse_v1 '--reg_mode hc' training.

Hard Concrete (Louizos et al., ICLR 2018 "Learning Sparse Neural Networks
through L0 Regularization") is a single-parameter stochastic binary gate that
produces exact 0s and 1s with nonzero probability while remaining
reparameterizable.

Parameterization (per kv-head scalar ``log_alpha``):
    u ~ Uniform(0, 1)                   [frozen per step, like kuma]
    s = sigmoid((log(u/(1-u)) + log_alpha) / beta)
    z_bar = s * (zeta - gamma) + gamma  [stretch beyond [0,1]]
    z = clamp(z_bar, 0, 1)              [hard rectify -> exact 0 and 1]

where beta=2/3, gamma=-0.1, zeta=1.1 (Louizos et al. defaults).

Key properties vs HardKuma:
  * Single parameter (log_alpha): 1-D optimization landscape, no (a,b) coupling.
  * P(z > 0) gradient: d/d(log_alpha) sigmoid(...) is always nonzero and is
    MAXIMIZED at log_alpha = -beta*log(-gamma/zeta) (the "middle" head) -- the
    opposite of the kuma (a≈b≈1) dead zone.
  * L0 penalty = sum P(z_h > 0) with a FIXED --reg_weight (no Lagrangian).
  * Export: log_alpha is monotone with P(z > 0), so global top-k on log_alpha
    hits the exact target_sparsity.

Noise convention: u is drawn ONCE per step with a rank-shared seed (outside all
forward / AC regions) and stored in holder.hc_noise, exactly mirroring the kuma
noise protocol. z is recomputed INSIDE each layer forward from self.log_alpha +
frozen u, so the gradient stays in the FSDP-hooked forward.
"""

import math
import torch

# Hard Concrete hyper-parameters (Louizos et al. defaults).
HC_BETA  = 2.0 / 3.0   # temperature
HC_GAMMA = -0.1         # stretch lower bound
HC_ZETA  =  1.1         # stretch upper bound
# Uniform noise clamp (same as kuma NOISE_EPS).
NOISE_EPS = 1e-3


def hc_sample_z(u, log_alpha, beta=HC_BETA, gamma=HC_GAMMA, zeta=HC_ZETA):
    """Reparameterize a Hard Concrete sample from frozen uniform noise ``u``.

    z = clamp( sigmoid((logit(u) + log_alpha) / beta) * (zeta-gamma) + gamma, 0, 1 )

    Gradient flows to ``log_alpha`` only (``u`` is a constant).
    Returns z in [0, 1] with nonzero probability mass at exactly 0 and 1.
    """
    # logit(u) = log(u / (1-u))
    logit_u = torch.log(u) - torch.log1p(-u)
    s = torch.sigmoid((logit_u + log_alpha) / beta)
    z_bar = s * (zeta - gamma) + gamma
    return z_bar.clamp(0.0, 1.0)


def hc_p_nonzero(log_alpha, beta=HC_BETA, gamma=HC_GAMMA, zeta=HC_ZETA):
    """P(z > 0) = sigmoid((log_alpha - beta * log(-gamma/zeta)) / 1).

    This is the Hard Concrete L0 regularizer term per gate (Louizos eq. 12).
    Gradient w.r.t. log_alpha = sigmoid' > 0 everywhere -- no dead zone.
    """
    shift = beta * math.log(-gamma / zeta)   # beta * log(0.1/1.1) < 0
    return torch.sigmoid(log_alpha - shift)


def deterministic_head_mask(log_alpha, target_sparsity, uniform_sparsity=False,
                            layer0_forced=True):
    """Deploy readout: zero the lowest-scoring heads to hit exact target sparsity.

    Returns a hard {0,1} mask of shape log_alpha.shape; 1 -> anchor (dense), 0 ->
    sparse. With ``uniform_sparsity`` each layer zeros the same per-layer count;
    otherwise a single global top-k over the eligible slots.

    ``layer0_forced=True``: layer 0 is always all-anchor, so the zeros are drawn
    from layers 1..L-1 only (``soft[0]`` is set to +inf before the top-k).
    ``layer0_forced=False``: layer 0 participates in the top-k competition on
    equal terms.
    """
    L, H = log_alpha.shape
    # Rank on the RAW log_alpha. It is monotone with P(z > 0) = sigmoid(log_alpha
    # - shift), so top-k on log_alpha == top-k on P(z > 0). Do NOT relu() first:
    # the L0 penalty drives most heads to log_alpha < 0, so relu would collapse
    # exactly the heads being ranked into one big tie at 0.0 and torch.topk would
    # then break the tie by flat index, i.e. by (layer, head) number instead of by
    # the learned value.
    soft = log_alpha.detach().clone().float()
    hard = torch.ones_like(soft)
    if uniform_sparsity:
        start_layer = 1 if layer0_forced else 0
        k = round(target_sparsity * H)
        for layer in range(start_layer, L):
            row = soft[layer].clone()
            marks = torch.ones(H, device=soft.device)
            if k > 0:
                _, idx = torch.topk(row, k=min(k, H), largest=False)
                marks[idx] = 0.0
            hard[layer] = marks
    else:
        if layer0_forced:
            soft[0] = float("inf")
        num_zeros = round(target_sparsity * L * H)
        flat = soft.reshape(-1).clone()
        if num_zeros > 0:
            _, idx = torch.topk(flat, k=min(num_zeros, flat.numel()), largest=False)
            hard.reshape(-1)[idx] = 0.0
    if layer0_forced:
        hard[0, :] = 1.0
    return hard
