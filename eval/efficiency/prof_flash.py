import os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from pbs_attn.patch.huggingface import apply_patch_with_prefill, get_flashattn_prefill

MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LEN = 128 * 1024

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model = apply_patch_with_prefill(model, get_flashattn_prefill())

vocab = model.get_input_embeddings().weight.size(0)
ids = torch.randint(0, vocab, (1, LEN), device=model.device, dtype=torch.long)

with torch.no_grad():
    for _ in range(5):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()

n = 5
t0 = time.time()
with torch.no_grad():
    for _ in range(n):
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
print(f"avg latency: {(time.time()-t0)/n:.4f}s")

from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
torch.cuda.synchronize()
print("\n===== TOP OPS BY CUDA TIME (flashattn) =====")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
