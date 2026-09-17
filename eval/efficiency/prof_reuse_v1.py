import os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from pbs_attn.patch.huggingface import apply_patch_with_prefill, get_reuse_v1_prefill

MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LEN = 128 * 1024

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
prefill_fn = get_reuse_v1_prefill(
    label_path=LABEL, select_mode="topp", top_p=0.9,
    min_blocks=16, max_blocks=64, per_head_topp=False, last_q_full=True,
    sink_blocks=1, local_blocks=2,
)
model = apply_patch_with_prefill(model, prefill_fn)

vocab = model.get_input_embeddings().weight.size(0)
ids = torch.randint(0, vocab, (1, LEN), device=model.device, dtype=torch.long)

# warmup
with torch.no_grad():
    for _ in range(5):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()

# timed
n = 5
t0 = time.time()
with torch.no_grad():
    for _ in range(n):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
print(f"avg latency: {(time.time()-t0)/n:.4f}s")

# profile one
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
print("\n===== TOP OPS BY CUDA TIME =====")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
