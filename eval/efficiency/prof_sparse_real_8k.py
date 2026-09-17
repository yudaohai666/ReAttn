"""Faithful 8k sparse-batching measurement: REAL selections + last_q_full.

1. Run ONE real LongBench-v2 8k prefill (our config: topp0.95/mb8/xb48/lastfull,
   fused_anchor=1). Capture every sparse `_sparse_block_attn` call's real k_sel/k_cnt
   (already reflects last_q_full: body = first nqb-1 q-blocks, last block -> flash).
2. Replay per layer: loop (per-kv-head) vs batched (all sparse kv-heads, one dim4
   launch), on the REAL selections. Report real cnt distribution + real E2E fraction.
3. Separately time the last_q_full flash last-block cost.
"""
import os, sys, math, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["REUSE_V1_FUSED_ANCHOR"] = "1"
os.environ.setdefault("http_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("https_proxy", "http://agent.baidu.com:8891")
REPO = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
sys.path.insert(0, REPO)

import pbs_attn.baselines.Reuse_v1 as R
from transformers import AutoTokenizer, AutoModelForCausalLM
from pbs_attn.patch.huggingface import apply_patch_with_prefill, get_reuse_v1_prefill
from datasets import load_dataset

MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL = f"{REPO}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LEN, DEV, SM = 8192, "cuda:0", 132
d = 128

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="eager").to(DEV).eval()
prefill_fn = get_reuse_v1_prefill(
    label_path=LABEL, budget=32, block_size=128, segment_size=2048,
    sink_blocks=1, local_blocks=2, causal=True,
    select_mode="topp", top_p=0.95, min_blocks=8, max_blocks=48,
    last_q_full=True, per_head_topp=False)
model = apply_patch_with_prefill(model, prefill_fn)

ds = load_dataset("THUDM/LongBench-v2", split="train")
ctx = ds[0]["context"]
ids = tok(ctx, return_tensors="pt").input_ids.to(DEV)
if ids.shape[1] < LEN:
    ids = ids.repeat(1, LEN // ids.shape[1] + 1)
ids = ids[:, :LEN]
print(f"real input tokens: {ids.shape[1]}")

# ---------- warmup so autotune settles ----------
with torch.no_grad():
    for _ in range(2):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()

# ---------- capture ONE real forward ----------
CAP = []  # ('L',) layer marker | ('S', G, qlen, s, k_sel, k_cnt)
orig_sparse = R._sparse_block_attn
orig_layer = R.reuse_v1_layer_per_hkv

def cap_sparse(*a, **kw):
    q = a[0]; ksel = kw.get("k_sel"); kcnt = kw.get("k_cnt")
    CAP.append(("S", q.shape[1], q.shape[2], a[2].shape[2], ksel, kcnt))
    return orig_sparse(*a, **kw)

def cap_layer(*a, **kw):
    CAP.append(("L",))
    return orig_layer(*a, **kw)

R._sparse_block_attn = cap_sparse
R.reuse_v1_layer_per_hkv = cap_layer
with torch.no_grad():
    model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
R._sparse_block_attn = orig_sparse
R.reuse_v1_layer_per_hkv = orig_layer

# ---------- group into layers ----------
layers, cur = [], None
for rec in CAP:
    if rec[0] == "L":
        if cur is not None:
            layers.append(cur)
        cur = []
    elif cur is not None:
        cur.append(rec)
if cur:
    layers.append(cur)
sparse_layers = [L for L in layers if L]
n_calls = sum(len(L) for L in sparse_layers)
print(f"layers total={len(layers)}  layers-with-sparse={len(sparse_layers)}  "
      f"sparse-head-calls={n_calls}")

# ---------- real cnt distribution ----------
allc = torch.cat([r[5].reshape(-1).float() for L in sparse_layers for r in L])
nkb = LEN // 128
print(f"real selected-blocks/q-block: mean={allc.mean():.1f} max={allc.max():.0f} "
      f"min={allc.min():.0f}  (cap max_blocks=48, nkb={nkb})")

# ---------- replay: loop vs batched, on REAL selections ----------
scale = 1.0 / math.sqrt(d)

def rand(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.bfloat16)

def build_layer(L):
    """Materialize per-head (loop) and stacked (batched) tensors for one layer."""
    b = 1
    G, qlen, s = L[0][1], L[0][2], L[0][3]
    max_sel = L[0][4].shape[-1]
    nqb_body = L[0][4].shape[-2] if L[0][4].dim() == 3 else L[0][4].shape[-2]
    loop = []
    for _, g, ql, ss, ksel, kcnt in L:
        loop.append((rand(b, g, ql, d), rand(b, 1, ss, d), rand(b, 1, ss, d), ksel, kcnt))
    N = len(L)
    qB = rand(b, N * G, qlen, d)
    kB = rand(b, N, s, d); vB = rand(b, N, s, d)
    # dim4 per-head sel: each kv-head's shared (b,nqb,max_sel) -> broadcast to its G q-heads
    selP = torch.empty(b, N * G, nqb_body, max_sel, dtype=torch.int32, device=DEV)
    cntP = torch.empty(b, N * G, nqb_body, dtype=torch.int32, device=DEV)
    for i, (_, g, ql, ss, ksel, kcnt) in enumerate(L):
        selP[:, i * G:(i + 1) * G] = ksel.unsqueeze(1)
        cntP[:, i * G:(i + 1) * G] = kcnt.unsqueeze(1)
    return loop, (qB, kB, vB, selP, cntP)

built = [build_layer(L) for L in sparse_layers]

def loop_all():
    for loop, _ in built:
        for q, k, v, ksel, kcnt in loop:
            orig_sparse(q, k, v, None, 48, 128, True, scale, k_sel=ksel, k_cnt=kcnt)

def batch_all():
    for _, (qB, kB, vB, selP, cntP) in built:
        orig_sparse(qB, kB, vB, None, 48, 128, True, scale, k_sel=selP, k_cnt=cntP)

def timed(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2]

t_loop = timed(loop_all)
t_batch = timed(batch_all)

# ---------- last_q_full flash cost (per sparse head, last block) ----------
from flash_attn import flash_attn_func
last_len = LEN - (nkb - 1) * 128
def flash_all():
    for L in sparse_layers:
        G, s = L[0][1], L[0][3]
        for _ in L:
            lq = rand(1, last_len, G, d); k = rand(1, s, 1, d); v = rand(1, s, 1, d)
            flash_attn_func(lq, k, v, softmax_scale=scale, causal=True)
t_flash = timed(flash_all, iters=10, warmup=3)

E2E = 0.346
print("\n================ REAL-DATA 8k sparse-path measurement ================")
print(f"sparse kernel  loop (per-hkv) : {t_loop:8.3f} ms")
print(f"sparse kernel  batched (A1)   : {t_batch:8.3f} ms   -> {t_loop/t_batch:.2f}x, "
      f"saves {t_loop-t_batch:.3f} ms")
print(f"last_q_full flash (all heads) : {t_flash:8.3f} ms   (NOT reduced by sparse batching)")
print(f"--- E2E fractions (8k prefill = {E2E*1000:.0f} ms) ---")
print(f"sparse kernel share : {100*t_loop/1000/E2E:.1f}%   flash-lastblk share : {100*t_flash/1000/E2E:.1f}%")
print(f"batching E2E saving : {100*(t_loop-t_batch)/1000/E2E:.2f}%  ({t_loop-t_batch:.2f} ms of {E2E*1000:.0f} ms)")


