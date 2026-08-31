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
  * Density = mean P(z_h > 0) = mean sigmoid((log_alpha + beta*log(-gamma/zeta))/1)
    drives the Lagrangian exactly like kuma's hardkuma_density.
  * Export: log_alpha > 0 <=> P(z > 0.5) > 0.5; use global top-k on log_alpha
    to hit exact target_sparsity (monotone with density).

Noise convention: u is drawn ONCE per step with a rank-shared seed (outside all
forward / AC regions) and stored in holder.hc_noise, exactly mirroring the kuma
noise protocol. z is recomputed INSIDE each layer forward from self.log_alpha +
frozen u, so the gradient stays in the FSDP-hooked forward.
"""

import math
import torch
from torch.nn import functional as F

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


def hc_density(log_alpha, beta=HC_BETA, gamma=HC_GAMMA, zeta=HC_ZETA):
    """Expected anchor (z>0) fraction: mean P(z_h > 0) over all gates.

    This is what the Lagrangian density constraint drives toward desired_density.
    """
    return hc_p_nonzero(log_alpha, beta, gamma, zeta).mean()


def hc_p_positive(log_alpha, beta=HC_BETA, gamma=HC_GAMMA, zeta=HC_ZETA):
    """P(z > 0.5): probability that the gate is in the anchor half.

    Used as an alternative density measure aligned with the z>0.5 export
    threshold (mirroring STG's Phi(mu/sigma) = P(z>0.5)).

    P(z > 0.5) = P(z_bar > 0.5) = P(s > (0.5 - gamma)/(zeta - gamma))
               = P(logit(u) + log_alpha > beta * logit((0.5-gamma)/(zeta-gamma)))
               = sigmoid( (log_alpha - beta * logit((0.5-gamma)/(zeta-gamma))) )
    """
    x = (0.5 - gamma) / (zeta - gamma)   # (0.5 - (-0.1)) / (1.1 - (-0.1)) = 0.5
    # x = 0.5 exactly with the default gamma/zeta, so logit(x) = 0.
    # -> P(z > 0.5) = sigmoid(log_alpha / beta * ... ) simplifies.
    logit_x = math.log(x / (1.0 - x))    # logit(0.5) = 0 for defaults
    return torch.sigmoid((log_alpha - beta * logit_x))


def hc_export_score(log_alpha):
    """Score for deterministic export: higher = more anchor.

    log_alpha is monotone with both hc_p_nonzero and hc_p_positive, so
    global top-k on log_alpha gives exactly target_sparsity anchor heads.
    Export threshold: log_alpha > 0 <=> P(z>0) > 0.5 (with default params,
    P(z>0.5) > 0.5 as well since logit(0.5)=0).
    """
    return log_alpha
