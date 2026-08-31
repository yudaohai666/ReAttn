"""Ulysses sequence parallelism primitive for reuse_v1 training.

The single reusable building block is a **differentiable 4-D all-to-all** that
converts between the two layouts of a multi-head tensor:

    seq-sharded / all-heads   (b, H,       s_local, d)
    head-sharded / full-seq   (b, H_local, s_full,  d)     s_full = s_local * P

with ``P = sp_size`` ranks in the sequence-parallel group and
``H_local = H // P``.  It is used INSIDE attention: project q/k/v on the local
sequence shard (all heads), all-to-all to "full sequence / local head subset",
run the reuse_v1 block-sparse kernels + gate blend on that head subset (which
sees the WHOLE sequence -- this is why global top-k block selection stays exact
under Ulysses, unlike Ring attention), then all-to-all back to "local sequence /
all heads" before ``o_proj``.

The all-to-all is wrapped in an ``autograd.Function`` whose backward is the
inverse all-to-all, so gradients flow across ranks automatically -- a label
position living on one rank propagates its gradient to the gate slice owned by
another rank through the collective's backward.

reuse_v1 is uniquely friendly to this: everything (block-sparse kernel, per-
block scores, top-k selection, the cross-layer ``holder.sels`` reuse, the
per-kv-head gate) is per-(layer, kv-head), so head-dim sharding is native and
the kernels need ZERO changes.  Because the head assignment is CONSTANT across
layers, layer L's selection stays LOCAL to the rank that will consume it at
layer L+1 -- no extra communication for the reuse threading.
"""

import torch
import torch.distributed as dist


class SPContext:
    """Sequence-parallel group metadata shared across all layers of a model.

    ``sp_size == 1`` means SP is disabled (single-rank / pure data-parallel);
    callers gate the SP code path on ``sp_size > 1`` so the non-SP path is
    byte-for-byte unchanged.
    """

    def __init__(self, group=None):
        self.group = group
        if group is None:
            self.sp_size = 1
            self.sp_rank = 0
        else:
            self.sp_size = dist.get_world_size(group)
            self.sp_rank = dist.get_rank(group)

    def head_range(self, num_heads):
        """[lo, hi) slice of ``num_heads`` owned by this rank after all-to-all."""
        assert num_heads % self.sp_size == 0, (
            f"num_heads={num_heads} not divisible by sp_size={self.sp_size}")
        local = num_heads // self.sp_size
        lo = self.sp_rank * local
        return lo, lo + local


def _all_to_all_4d(x, scatter_dim, gather_dim, group):
    """Raw (non-autograd) 4-D all-to-all between the two layouts.

    (scatter_dim=1, gather_dim=2): (b, H, s_local, d) -> (b, H_local, s_full, d)
        scatter the HEAD dim across ranks, gather the SEQ dim.
    (scatter_dim=2, gather_dim=1): (b, H_local, s_full, d) -> (b, H, s_local, d)
        the exact inverse.

    Sequence is sharded as CONTIGUOUS chunks (rank p holds global tokens
    [p*s_local, (p+1)*s_local)), so the gathered seq axis is in global order.
    """
    P = dist.get_world_size(group)
    if P == 1:
        return x

    if scatter_dim == 1 and gather_dim == 2:
        b, H, s_local, d = x.shape
        H_local = H // P
        # split head dim into P groups, bring the group axis to the front
        t = x.reshape(b, P, H_local, s_local, d).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(t)
        dist.all_to_all_single(out, t, group=group)
        # out[j] = source-rank-j's seq shard for THIS rank's head group
        out = out.permute(1, 2, 0, 3, 4).contiguous()  # (b, H_local, P, s_local, d)
        return out.reshape(b, H_local, P * s_local, d)

    if scatter_dim == 2 and gather_dim == 1:
        b, H_local, s_full, d = x.shape
        s_local = s_full // P
        # split seq dim into P shards, bring the shard axis to the front
        t = x.reshape(b, H_local, P, s_local, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(t)
        dist.all_to_all_single(out, t, group=group)
        # out[j] = source-rank-j's head group for THIS rank's seq shard
        out = out.permute(1, 0, 2, 3, 4).contiguous()  # (b, P, H_local, s_local, d)
        return out.reshape(b, P * H_local, s_local, d)

    raise ValueError(f"unsupported (scatter_dim={scatter_dim}, gather_dim={gather_dim})")


class _SeqAllToAll(torch.autograd.Function):
    """Differentiable 4-D all-to-all; backward is the inverse all-to-all."""

    @staticmethod
    def forward(ctx, group, x, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all_4d(x, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = _all_to_all_4d(
            grad_out.contiguous(), ctx.gather_dim, ctx.scatter_dim, ctx.group)
        return (None, grad_in, None, None)


def seq_to_head(x, group):
    """(b, H, s_local, d) -> (b, H_local, s_full, d). Differentiable."""
    return _SeqAllToAll.apply(group, x, 1, 2)


def head_to_seq(x, group):
    """(b, H_local, s_full, d) -> (b, H, s_local, d). Differentiable."""
    return _SeqAllToAll.apply(group, x, 2, 1)
