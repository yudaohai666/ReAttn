"""Hard Logistic (HL) gate for reuse_v1 '--reg_mode hln' training.

HardLogistic(mu, s) is Hard Concrete (Louizos et al. ICLR 2018) with the
fixed temperature beta replaced by a global learnable temperature s > 0.

Construction (per kv-head scalar mu, global temperature s):
    u     ~ Uniform(0, 1)                      [frozen per step, like hc]
    t      = sigmoid((logit(u) + mu) / s)       [logistic noise reparameterization]
    z_bar  = t * (zeta - gamma) + gamma         [stretch beyond [0,1]]
    z      = clamp(z_bar, 0, 1)                [hard rectify -> exact 0 and 1]

where gamma=-0.1, zeta=1.1 (same HC defaults, zeta+gamma=1.0).

This is structurally identical to HC; s plays the role of beta. Because
zeta+gamma=1.0 the midpoint threshold simplifies exactly:

    P(z > 0.5) = P(z_bar > 0.5) = P(t > 0.5)
               = P(sigmoid((logit(u)+mu)/s) > 0.5)
               = P((logit(u)+mu)/s > 0)
               = P(logit(u) > -mu)
               = sigmoid(mu)              <- s cancels exactly

So density = sigmoid(mu) regardless of s. s controls only gate sharpness
(small s -> hard gate; large s -> soft/exploratory). s cannot hijack the
Lagrangian because it does not appear in the density.

Noise convention: u is drawn ONCE per step with a rank-shared seed (outside
all forward / AC regions) and stored in holder.hln_noise, exactly mirroring
the hc noise protocol (uniform, not Gaussian). z is recomputed INSIDE each
layer forward from (mu, s, frozen u), so the mu gradient stays in the
FSDP-hooked forward and AC recompute deterministically reproduces the same z.

Key properties vs HardKuma:
  * No (a,b) dead zone: d/dmu sigmoid(mu)|_{mu=0} = 0.25 (maximum, never 0).
  * s-free density: sigmoid(mu) contains no s; Lagrangian drives mu only.
  * Export: mu > 0  <=>  sigmoid(mu) > 0.5  -- exact alignment.
  * s is global (not per-head): prevents per-head s from hijacking sparsity.
  * s constrained: s = softplus(s_raw) + HL_S_MIN to prevent collapse to 0.

Key properties vs HC (single-parameter):
  * s learnable -> gate sharpness adapts during training (warm exploration,
    convergence hardening) without manual beta tuning.
  * density stays sigmoid(mu): adding s does not break Lagrangian alignment.

Density for Lagrangian:
    density(mu) = sigmoid(mu)   [per-head, s-free]
    mean_density = mean_{l,h} sigmoid(mu_{l,h})

Export (deterministic binary label):
    z = 1[mu > 0]    (sigmoid(mu) > 0.5  <=>  mu > 0)
    top-k on mu gives the same ranking -> reuse deterministic_head_mask.
"""

import torch
import torch.nn.functional as F

# HC stretch parameters (same defaults so zeta+gamma=1.0 -> density s-free).
HL_GAMMA   = -0.1
HL_ZETA    =  1.1
# Minimum temperature to prevent s -> 0 collapse.
HL_S_MIN   = 0.05
# Uniform noise clamp (same as hc/kuma NOISE_EPS).
NOISE_EPS  = 1e-3


def hl_sample_z(u, mu, s, gamma=HL_GAMMA, zeta=HL_ZETA):
    """Reparameterize a HardLogistic sample from frozen uniform noise u.

    z = clamp( sigmoid((logit(u) + mu) / s) * (zeta-gamma) + gamma, 0, 1 )

    Gradient flows to mu only (u and s are constants at call time).
    Returns z in [0, 1] with nonzero probability mass at exactly 0 and 1.

    Args:
        u:   (num_layers, Hkv) frozen Uniform(0,1) noise drawn once per step,
             clamped away from {0,1} by NOISE_EPS before storage.
        mu:  (Hkv,) or (num_layers, Hkv) learnable location parameter.
        s:   float or scalar tensor -- temperature (positive, >= HL_S_MIN).
    """
    logit_u = torch.log(u) - torch.log1p(-u)   # logit(u)
    t = torch.sigmoid((logit_u + mu) / s)
    z_bar = t * (zeta - gamma) + gamma
    return z_bar.clamp(0.0, 1.0)


def hl_density(mu):
    """Expected anchor fraction per head: sigmoid(mu).

    P(z > 0.5) = sigmoid(mu)   [exact, s-free; derivation in module docstring]

    Full tensor, shape matching mu.  Used directly in the Lagrangian:
        c0 = mean(hl_density(mu)) - desired_density
    """
    return torch.sigmoid(mu)


def hl_density_mean(mu):
    """Mean expected anchor fraction over all (layer, head) slots."""
    return hl_density(mu).mean()


def hl_s_from_raw(s_raw):
    """Map unconstrained s_raw -> positive temperature s >= HL_S_MIN.

    s = softplus(s_raw) + HL_S_MIN

    Register s_raw as nn.Parameter; compute s via this function before
    passing to hl_sample_z and storing in holder.hln_sigma for logging.
    """
    return F.softplus(s_raw) + HL_S_MIN


def hl_export_score(mu):
    """Score for deterministic export: higher mu = more anchor.

    mu is monotone with sigmoid(mu), so top-k on mu gives exactly
    target_sparsity anchor heads.
    Export threshold: mu > 0  <=>  sigmoid(mu) > 0.5.
    """
    return mu
