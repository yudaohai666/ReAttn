"""Bit-identical check: GQA-fused sparse path vs production `_block_sparse_indexed_fwd`.

Run under BLOCK_SPARSE_NO_AUTOTUNE=1 so production uses its fixed BLOCK_N=128 config,
matching the fused kernel's BLOCK_N=128. With head-shared selection (nohead), the two
must produce a bit-identical `out` (per-row online softmax is independent; BLOCK_M only
tiles the stacked rows). Then a quick speed check at 128K single kv-group.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import math
import torch

DEV = "cuda"


def build_headshared_selection(b, nqb, nkb, cap, device, seed=0):
    """Random causal head-shared block selection -> (k_sel (b,nqb,max_sel), k_cnt (b,nqb))."""
    from pbs_attn.baselines.Reuse_v1 import _compact_block_mask
    g = torch.Generator(device=device).manual_seed(seed)
    qb = torch.arange(nqb, device=device)[:, None]
    kb = torch.arange(nkb, device=device)[None, :]
    causal = kb <= qb                                   # (nqb, nkb)
    rand = torch.rand((b, nqb, nkb), device=device, generator=g)
    keep = (rand < (cap / max(1, nkb))) & causal[None]  # sparse subset
    keep[:, :, 0] = True                                # sink
    keep[:, qb.flatten(), qb.flatten()] = True          # diagonal
    keep &= causal[None]
    max_sel = min(cap + 2, nkb)
    return _compact_block_mask(keep, max_sel)


def check(s=2048, Hkv=2, G=4, block_size=128, cap=16):
    from pbs_attn.baselines._sparse_block_attn_import import _sparse_block_attn  # placeholder
    pass


def run_check(s=2048, Hkv=2, G=4, block_size=128, cap=16):
    from pbs_attn.baselines.Reuse_v1 import _sparse_block_attn
    H = Hkv * G
    d = 128
    nqb = s // block_size
    nkb = nqb
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(b_ := 1, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    k_sel, k_cnt = build_headshared_selection(1, nqb, nkb, cap, DEV)

    os.environ["REUSE_V1_FUSED_SPARSE"] = "0"
    ref = _sparse_block_attn(q, k, v, None, cap, block_size, True, scale,
                             k_sel=k_sel, k_cnt=k_cnt)
    os.environ["REUSE_V1_FUSED_SPARSE"] = "1"
    got = _sparse_block_attn(q, k, v, None, cap, block_size, True, scale,
                             k_sel=k_sel, k_cnt=k_cnt)

    diff = (ref.float() - got.float()).abs().max().item()
    ident = torch.equal(ref, got)
    print(f"[check s={s} Hkv={Hkv} G={G} cap={cap}]  max_abs_diff={diff:.3e}  bit-identical={ident}")
    return ident


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


def bench(s=128 * 1024, G=4, block_size=128, cap=64):
    from pbs_attn.baselines.Reuse_v1 import _sparse_block_attn
    Hkv = 1
    H = Hkv * G
    d = 128
    nqb = s // block_size
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(1, H, s, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, s, d, device=DEV, dtype=torch.bfloat16)
    k_sel, k_cnt = build_headshared_selection(1, nqb, nqb, cap, DEV)

    os.environ["REUSE_V1_FUSED_SPARSE"] = "0"
    tp = timed(lambda: _sparse_block_attn(q, k, v, None, cap, block_size, True, scale,
                                          k_sel=k_sel, k_cnt=k_cnt))
    os.environ["REUSE_V1_FUSED_SPARSE"] = "1"
    tf = timed(lambda: _sparse_block_attn(q, k, v, None, cap, block_size, True, scale,
                                          k_sel=k_sel, k_cnt=k_cnt))
    print(f"[bench s={s//1024}K G={G} cap={cap}]  prod {tp:.3f}ms  fused {tf:.3f}ms  ({tp/tf:.2f}x)")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--check":
        assert os.environ.get("BLOCK_SPARSE_NO_AUTOTUNE", "false").lower() in ("1", "true", "yes"), \
            "run --check under BLOCK_SPARSE_NO_AUTOTUNE=1 so production uses BLOCK_N=128"
        ok = True
        for cap in (8, 16, 32):
            for (Hkv, G) in ((1, 4), (2, 4)):
                ok &= run_check(s=2048, Hkv=Hkv, G=G, cap=cap)
        print("CORRECTNESS:", "PASS (bit-identical)" if ok else "FAIL")
    else:
        assert os.environ.get("BLOCK_SPARSE_NO_AUTOTUNE", "false").lower() not in ("1", "true", "yes"), \
            "run --bench with autotune ON (do NOT set BLOCK_SPARSE_NO_AUTOTUNE)"
        for cap in (32, 64):
            bench(s=128 * 1024, G=4, cap=cap)
