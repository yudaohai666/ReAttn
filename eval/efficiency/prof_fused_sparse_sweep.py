"""GQA-fused-sparse vs default-loop vs batched, across lengths, on REAL selections.

Per length: run one real LongBench-v2 forward, capture the real per-layer sparse
selections (k_sel/k_cnt), then replay the sparse path three ways on those real
selections:
  - default : per-hkv loop, `_block_sparse_indexed_fwd` (autotuned) [production]
  - fused   : per-hkv loop, `_gqa_fused_indexed_fwd` (REUSE_V1_FUSED_SPARSE=1)
  - batched : all sparse kv-heads of a layer folded into ONE dim4 launch (A1)
Shared q/k/v buffers (values don't affect timing) so 128k doesn't OOM.
"""
import os, sys, math, torch
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
DEV, d, G = "cuda:0", 128, 4
LENGTHS = [8192, 16384, 32768, 65536, 131072]
scale = 1.0 / math.sqrt(d)

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
base_ids = tok(ds[0]["context"], return_tensors="pt").input_ids.to(DEV)


def capture(ids):
    """One real forward; return per-layer list of (k_sel, k_cnt) for sparse heads."""
    CAP = []
    os_sparse = R._sparse_block_attn
    os_layer = R.reuse_v1_layer_per_hkv

    def cap_sparse(*a, **kw):
        CAP.append(("S", kw.get("k_sel"), kw.get("k_cnt")))
        return os_sparse(*a, **kw)

    def cap_layer(*a, **kw):
        CAP.append(("L",))
        return os_layer(*a, **kw)

    R._sparse_block_attn = cap_sparse
    R.reuse_v1_layer_per_hkv = cap_layer
    with torch.no_grad():
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    R._sparse_block_attn = os_sparse
    R.reuse_v1_layer_per_hkv = os_layer

    layers, cur = [], None
    for rec in CAP:
        if rec[0] == "L":
            if cur:
                layers.append(cur)
            cur = []
        elif cur is not None:
            cur.append((rec[1], rec[2]))
    if cur:
        layers.append(cur)
    return [L for L in layers if L]


def timed(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


def warmup_fwd(ids):
    with torch.no_grad():
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()


print(f"\n{'len':>5} {'cnt_mean':>8} {'sp_calls':>8} "
      f"{'default_ms':>10} {'fused_ms':>9} {'batched_ms':>10} "
      f"{'fused/def':>10} {'batch/def':>10}")
for L in LENGTHS:
    ids = base_ids
    if ids.shape[1] < L:
        ids = ids.repeat(1, L // ids.shape[1] + 1)
    ids = ids[:, :L].contiguous()
    s = L
    nqb = L // 128
    qlen = (nqb - 1) * 128            # last_q_full body length

    warmup_fwd(ids)
    layers = capture(ids)
    calls = [c for lay in layers for c in lay]
    max_sel = calls[0][0].shape[-1]
    cnt_mean = torch.cat([c[1].reshape(-1).float() for c in calls]).mean().item()
    n_calls = len(calls)
    maxN = max(len(lay) for lay in layers)

    # shared buffers (values irrelevant to timing)
    qG = torch.randn(1, G, qlen, d, device=DEV, dtype=torch.bfloat16)
    k1 = torch.randn(1, 1, s, d, device=DEV, dtype=torch.bfloat16)
    v1 = torch.randn(1, 1, s, d, device=DEV, dtype=torch.bfloat16)
    qB = torch.randn(1, maxN * G, qlen, d, device=DEV, dtype=torch.bfloat16)
    kB = torch.randn(1, maxN, s, d, device=DEV, dtype=torch.bfloat16)
    vB = torch.randn(1, maxN, s, d, device=DEV, dtype=torch.bfloat16)
    selP = torch.zeros(1, maxN * G, qlen // 128, max_sel, dtype=torch.int32, device=DEV)
    cntP = torch.zeros(1, maxN * G, qlen // 128, dtype=torch.int32, device=DEV)

    def loop():
        for ksel, kcnt in calls:
            R._sparse_block_attn(qG, k1, v1, None, 48, 128, True, scale, k_sel=ksel, k_cnt=kcnt)

    def batched():
        for lay in layers:
            N = len(lay)
            for i, (ksel, kcnt) in enumerate(lay):
                selP[:, i * G:(i + 1) * G] = ksel.unsqueeze(1)
                cntP[:, i * G:(i + 1) * G] = kcnt.unsqueeze(1)
            R._sparse_block_attn(qB[:, :N * G], kB[:, :N], vB[:, :N], None, 48, 128,
                                 True, scale, k_sel=selP[:, :N * G], k_cnt=cntP[:, :N * G])

    os.environ["REUSE_V1_FUSED_SPARSE"] = "0"
    t_def = timed(loop)
    os.environ["REUSE_V1_FUSED_SPARSE"] = "1"
    t_fused = timed(loop)
    os.environ["REUSE_V1_FUSED_SPARSE"] = "0"
    t_batch = timed(batched)

    print(f"{L//1024:>4}k {cnt_mean:>8.1f} {n_calls:>8} "
          f"{t_def:>10.3f} {t_fused:>9.3f} {t_batch:>10.3f} "
          f"{t_fused/t_def:>9.2f}x {t_batch/t_def:>9.2f}x")
    del qG, k1, v1, qB, kB, vB, selP, cntP, calls, layers
    torch.cuda.empty_cache()


