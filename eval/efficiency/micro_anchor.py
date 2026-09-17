import torch, math
from pbs_attn.baselines.Reuse_v1 import (
    block_sparse_attn_with_score, block_sparse_attn_dense_no_score,
)
from flash_attn import flash_attn_func

torch.manual_seed(0)
b, G, s, d = 1, 4, 128 * 1024, 128
bs, seg = 128, 2048
dev = "cuda"
scale = 1.0 / math.sqrt(d)

q = torch.randn(b, G, s, d, device=dev, dtype=torch.bfloat16)
k = torch.randn(b, 1, s, d, device=dev, dtype=torch.bfloat16)
v = torch.randn(b, 1, s, d, device=dev, dtype=torch.bfloat16)
nqb = s // bs
full_bm = torch.ones((b, G, nqb, nqb), dtype=torch.bool, device=dev)

# flash layout: (b, seqlen, nheads, d); K/V with 1 kv-head (GQA)
qf = q.permute(0, 2, 1, 3).contiguous()
kf = k.permute(0, 2, 1, 3).contiguous()
vf = v.permute(0, 2, 1, 3).contiguous()

def timed(fn, n=20, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(n):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n

A = timed(lambda: block_sparse_attn_with_score(q, k, v, full_bm, block_size=bs, segment_size=seg, causal=True, softmax_scale=scale, q_chunk_blocks=None))
B = timed(lambda: block_sparse_attn_dense_no_score(q, k, v, block_size=bs, segment_size=seg, causal=True, softmax_scale=scale))
C = timed(lambda: flash_attn_func(qf, kf, vf, softmax_scale=scale, causal=True))

print(f"\n=== anchor kernel decomposition (b={b} G={G} s={s} d={d}) ===")
print(f"A  with_score (EMIT_SCORE=True + reduce) : {A:.3f} ms   <- current anchor path")
print(f"B  dense_no_score (same branch, no score): {B:.3f} ms")
print(f"C  flash_attn_func (floor)               : {C:.3f} ms")
print(f"--- decomposition ---")
print(f"flash floor                = {C:.3f} ms")
print(f"branch/codegen gap (B - C) = {B - C:.3f} ms")
print(f"scoring overhead   (A - B) = {A - B:.3f} ms")
