"""Stochastic Gates (STG) for reuse_v1 '--reg_mode stg' training.

Port of the official runopti/stg FeatureSelector gate machinery to the
per-(layer, kv-head) head-selection setting.

Reference: "Feature Selection using Stochastic Gates" (ICML 2020)
           https://github.com/runopti/stg

Gate polarity matches kuma_gate / ddp_gate: z=1 -> anchor (dense), z=0 -> sparse.

Official parameterization (code, not paper):
  mu_code ~ N(0, 0.01)   (stored as anchor_heads, same slot as ddp z_loga)
  mu_paper = mu_code + 0.5   (paper uses mu_paper ~ 0.5 initially)

Training sample:
  z = clip(mu_code + 0.5 + sigma * epsilon,  0, 1)
  epsilon ~ N(0,1), drawn once per step, frozen across fill/use passes.

Expected gate-active (anchor) probability per head:
  p(z > 0.5) = Phi(mu_code / sigma)   [Gaussian CDF]

  Note: Phi(mu/sigma) = P(mu + 0.5 + sigma*eps > 0.5) = P(eps > -mu/sigma)
  This tracks "gate majority-vote anchor" rather than "gate ever non-zero".
  At the sparse pole (mu << -0.5), Phi(mu/sigma) -> 0 much faster than
  Phi((mu+0.5)/sigma), so the Lagrangian constraint accurately tracks the
  true anchor-head count and aligns with the top-k export threshold mu > 0
  (Phi(0/sigma) = 0.5, i.e. gate median = 0.5).

Density (mean anchor fraction):
  density = mean_{l,h}  Phi(mu_{l,h} / sigma)

Used for the target-density Lagrangian:
  c = density - desired_density
  reg_loss = lambda * c   (single learnable lambda, dual-ascent maximize=True)

Export (deterministic binary label):
  mu_code values are monotone with Phi(mu/sigma), so top-k on mu_code
  gives the same ranking -> reuse deterministic_head_mask from ddp_gate.py.
  Threshold export: mu > 0  <=>  Phi(mu/sigma) > 0.5  (gate median > 0.5).
"""

import math
import torch

# Noise clamp to keep epsilon away from extreme values (mirrors kuma NOISE_EPS)
NOISE_EPS = 1e-3


def stg_sample_z(mu, sigma, noise):
    """Sample stochastic gate z from frozen per-step Gaussian noise.

    Args:
        mu:    (num_layers, Hkv) float tensor -- the learnable logit (mu_code).
        sigma: float -- noise scale (hyperparameter, default 0.5).
        noise: (num_layers, Hkv) float tensor -- frozen N(0,1) noise drawn once
               per step outside any AC region (same pattern as kuma_noise).

    Returns:
        z: (num_layers, Hkv) float tensor in [0, 1].
           Layer 0 should be forced all-anchor by the caller (same as ddp/kuma).
    """
    return torch.clamp(mu + 0.5 + sigma * noise, 0.0, 1.0)


def stg_density(mu, sigma):
    """Expected anchor fraction per head: Phi(mu / sigma).

    This is P(z > 0.5) = P(mu + 0.5 + sigma*eps > 0.5) = P(eps > -mu/sigma)
    for eps ~ N(0,1).  At the sparse pole (mu << -0.5) this approaches 0
    much faster than Phi((mu+0.5)/sigma), so the Lagrangian constraint tracks
    the true anchor-head count more accurately and aligns with the top-k
    export threshold (mu > 0 <=> Phi(mu/sigma) > 0.5).
    Full tensor, shape (num_layers, Hkv).
    """
    x = mu / sigma
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def stg_density_mean(mu, sigma):
    """Mean expected anchor fraction over all (layer, head) slots."""
    return stg_density(mu, sigma).mean()
