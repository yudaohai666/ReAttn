"""E2E reuse_v1 prefill profile on a SINGLE GPU (no accelerate/flash needed).

Loads Llama-3.1-8B to cuda:0 directly (device_map avoided so accelerate isn't
required), patches in the reuse_v1 prefill, times prefill at PROF_LEN, and prints
a CUDA-kernel profiler table so the anchor (with_score+reduce) vs sparse-path share
is visible. Used to compare before/after the GQA-fused anchor port.
"""
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pbs_attn.patch.huggingface import apply_patch_with_prefill, get_reuse_v1_prefill

REPO = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL = f"{REPO}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LEN = int(os.environ.get("PROF_LEN", str(128 * 1024)))
TAG = os.environ.get("PROF_TAG", "baseline")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="eager")
model = model.to("cuda:0").eval()

prefill_fn = get_reuse_v1_prefill(
    label_path=LABEL, budget=32, block_size=128, segment_size=2048,
    sink_blocks=1, local_blocks=2, causal=True,
    select_mode="topp", top_p=0.9, min_blocks=16, max_blocks=64,
    last_q_full=True, per_head_topp=False,
)
model = apply_patch_with_prefill(model, prefill_fn)

vocab = model.get_input_embeddings().weight.size(0)
ids = torch.randint(0, vocab, (1, LEN), device=model.device, dtype=torch.long)

with torch.no_grad():
    for _ in range(3):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()

N = 5
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
print(f"[timing TAG={TAG}] LEN={LEN} avg prefill = {(time.time()-t0)/N:.4f}s")

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# --- categorize CUDA time: attention kernels / projections / SELECTION / other ---
evs = prof.key_averages()
def cuda_us(e):
    return getattr(e, "self_cuda_time_total", getattr(e, "self_device_time_total", 0))
total = sum(cuda_us(e) for e in evs)
buckets = {"anchor": 0.0, "sparse": 0.0, "mm/gemm": 0.0, "selection": 0.0, "other": 0.0}
SEL_KEYS = ("topk", "sort", "scatter", "masked_fill", "cumsum", "gather",
            "block_score_reduce", "arange", "bitwise", "__and__", "__or__",
            "nonzero", "cumulative", "where")
for e in evs:
    n = e.key.lower()
    t = cuda_us(e)
    if "score_fwd" in n or "score_reduce" in n:
        buckets["anchor"] += t
    elif "sparse_indexed" in n or "gqa_fused_indexed" in n:
        buckets["sparse"] += t
    elif "nvjet" in n or n in ("aten::mm", "aten::bmm"):
        buckets["mm/gemm"] += t
    elif any(k in n for k in SEL_KEYS):
        buckets["selection"] += t
    else:
        buckets["other"] += t
print("\n=== CUDA time by category (self, ms) ===")
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"  {k:12s}: {v/1e3:8.1f} ms  ({100*v/total:4.1f}%)")
print(f"  {'TOTAL':12s}: {total/1e3:8.1f} ms")
print("\n=== selection-related ops (self CUDA ms) ===")
for e in sorted(evs, key=lambda e: -cuda_us(e)):
    n = e.key.lower()
    if any(k in n for k in SEL_KEYS) and cuda_us(e) > 0:
        print(f"  {e.key:45s}: {cuda_us(e)/1e3:7.2f} ms  x{e.count}")
