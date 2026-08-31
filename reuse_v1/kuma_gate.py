"""Minimal HardKuma gate for reuse_v1 '--reg_mode kuma' training.

A pared-down port of LycheeDecode's ``kuma.py`` (HardKuma / StretchedVariable /
Kuma) exposing only what the reuse_v1 two-way trainer needs:

  * ``hardkuma_sample_z(u, a, b, support)`` -- reparameterize a HardKuma sample
    z in [0,1] from a FROZEN uniform noise ``u`` and the shape params (a, b).
    This is the crux of the FSDP-correct integration: the noise is drawn ONCE
    per step (outside every forward / checkpoint region), and z is recomputed
    INSIDE each layer's blend from ``self.kuma_a/self.kuma_b`` -- so (a, b) stay
    consumed inside the FSDP-hooked layer forward (per-layer reduce-scatter
    AVERAGE), and the distill ``* world_size`` recipe compensates unchanged.
    Under activation-checkpoint recompute, the same frozen ``u`` deterministically
    reproduces z, so no resampling occurs.

  * ``hardkuma_density(a, b, support)`` -- E[z != 0] = mean(1 - pdf(0)); the
    quantity the Lagrangian density constraint drives toward desired_density.

  * ``hardkuma_mean(a, b)`` -- Kuma 1st moment (lgamma); used at export
    (``mean > 0.5`` -> anchor head), mirroring the reference.

Polarity matches the deterministic reuse_v1 gate: z=1 -> anchor (dense), z=0 ->
sparse (cheap). density = expected fraction of anchor heads.
"""

import math

import torch
from torch.nn import functional as F

EPS = 1e-6
# HardKuma stretch support (Louizos et al.); matches the reference train_kuma.
KUMA_SUPPORT = (-0.1, 1.1)
# Uniform noise clamp for the inverse-CDF reparam (reference Kuma.sample eps).
NOISE_EPS = 1e-3


def _kuma_log_cdf(x, a, b):
    """log CDF of a base Kuma(a,b) at x in (0,1). Matches Kuma.log_cdf."""
    r = 1.0 - (1.0 - x ** a) ** b
    r = torch.log(r + EPS)
    return r.clamp(math.log(EPS), math.log(1 - EPS))


def hardkuma_sample_z(u, a, b, support=KUMA_SUPPORT):
    """Reparameterize a HardKuma sample from frozen uniform noise ``u``.

    z = hardtanh( loc + scale * (1 - (1-u)^(1/b))^(1/a) , 0, 1 )

    ``u``, ``a``, ``b`` broadcast to the same shape; gradient flows to (a, b)
    only (``u`` is a constant). Returns z in [0, 1].
    """
    loc = support[0]
    scale = support[1] - support[0]
    x = (1.0 - (1.0 - u) ** b.reciprocal()) ** a.reciprocal()  # base Kuma
    y = x * scale + loc                                        # stretched
    return F.hardtanh(y, 0.0, 1.0)                             # hard-rectified


def hardkuma_pdf0(a, b, support=KUMA_SUPPORT):
    """HardKuma point mass at 0 = StretchedVariable.cdf(0) = exp(log_cdf(0))."""
    loc = support[0]
    scale = support[1] - support[0]
    x_ = (0.0 - loc) / scale  # shrink the stretched variable at x=0
    x0 = a.new_full((), float(x_))
    return _kuma_log_cdf(x0, a, b).exp()


def hardkuma_density(a, b, support=KUMA_SUPPORT):
    """Expected non-zero (anchor) fraction: mean(1 - pdf(0)) over all gates."""
    return (1.0 - hardkuma_pdf0(a, b, support)).mean()


def kuma_moments(a, b, n=1):
    """nth moment of a base Kuma(a,b) via lgamma (matches the reference)."""
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


def hardkuma_mean(a, b):
    """Kuma 1st moment; the export decision statistic (mean > 0.5 -> anchor)."""
    return kuma_moments(a, b, 1)
