"""E2E prefill timing: reuse_v1 (my method) vs pbs, across sequence lengths.

Single-GPU (no accelerate/flash): loads Llama-3.1-8B to cuda:0, patches in one
prefill method (METHOD env: 'reuse_v1' or 'pbs'), times prefill at each length in
LENS. reuse_v1 uses the eval config (topp tp=0.75, mb16, xb1024, nohead, last_q_full);
set REUSE_V1_FUSED_ANCHOR=1 to use the GQA-fused anchor kernel. pbs uses its default
config (block_size=128, segment_size=256, threshold=0.9). Run once per method.
"""
import os, time, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pbs_attn.patch.huggingface import (
    apply_patch_with_prefill, get_reuse_v1_prefill, get_permuted_block_sparse_attn_fwd)

REPO = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL = f"{REPO}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"

METHOD = os.environ.get("METHOD", "reuse_v1")
LENS = [int(x) * 1024 for x in os.environ.get("LENS", "8,16,32,64,128").split(",")]
WARM = int(os.environ.get("WARM", "2"))
N = int(os.environ.get("N", "3"))

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
_attn_impl = "eager"
if METHOD == "flash":
    _attn_impl = "sdpa"
elif METHOD == "flash2":
    _attn_impl = "flash_attention_2"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation=_attn_impl).to("cuda:0").eval()

if METHOD in ("flash", "flash2"):
    tag = f"{METHOD}({_attn_impl}-dense)"
    prefill_fn = None
elif METHOD == "reuse_v1":
    prefill_fn = get_reuse_v1_prefill(
        label_path=LABEL, budget=32, block_size=128, segment_size=2048,
        sink_blocks=1, local_blocks=2, causal=True,
        select_mode="topp", top_p=0.9, min_blocks=16, max_blocks=64,
        last_q_full=True, per_head_topp=False)
    tag = f"reuse_v1(fused={os.environ.get('REUSE_V1_FUSED_ANCHOR','0')})"
elif METHOD == "pbs":
    prefill_fn = get_permuted_block_sparse_attn_fwd(
        block_size=128, segment_size=256, threshold=0.9,
        causal=True, force_select_first_block=True, use_triton=True)
    tag = "pbs"
else:
    raise SystemExit(f"unknown METHOD={METHOD}")

if prefill_fn is not None:
    model = apply_patch_with_prefill(model, prefill_fn)
vocab = model.get_input_embeddings().weight.size(0)

def bench(LEN):
    ids = torch.randint(0, vocab, (1, LEN), device=model.device, dtype=torch.long)
    with torch.no_grad():
        for _ in range(WARM):
            model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()
    return (time.time() - t0) / N

res = {}
for LEN in LENS:
    dt = bench(LEN)
    res[LEN] = dt
    print(f"[{tag}] LEN={LEN//1024}K  prefill = {dt:.4f}s", flush=True)

print("RESULT_JSON " + json.dumps({"tag": tag, "method": METHOD,
      "fused": os.environ.get("REUSE_V1_FUSED_ANCHOR", "0"),
      "times": {str(k): v for k, v in res.items()}}))
