import sys; sys.path.insert(0, 'reuse_v1')
import torch
from ddp_gate import ste_relu, ste_clamp, soft_saturation_score, anneal_mean, deterministic_head_mask

x = torch.tensor([-0.5, 0.3, 1.7], requires_grad=True)
y = ste_clamp(x, 0, 1); print('ste_clamp fwd', y.tolist())
y.sum().backward(); print('ste_clamp grad', x.grad.tolist())

L, H = 4, 8
z = torch.randn(L, H) * 0.1 + 1.0
s = soft_saturation_score(z, anneal_mean(0.5))
print('score shape', tuple(s.shape), 'layer0 all-one', bool((s[0] == 1).all()),
      'range', round(float(s.min()), 4), round(float(s.max()), 4))

print('anneal p=0', anneal_mean(0.0), 'p=1', round(anneal_mean(1.0), 4))

z2 = torch.randn(4, 8)
hard = deterministic_head_mask(z2, 0.5)
print('mask layer0 all-anchor', bool((hard[0] == 1).all()),
      'zeros', int((hard == 0).sum().item()), 'expected', round(0.5 * 4 * 8))
print('OK')
