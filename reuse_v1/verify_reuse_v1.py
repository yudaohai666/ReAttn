"""Verify the reuse_v1 HF prefill-patch path on H800 (pure Triton).

Tier A: fast smoke without a model -- random q/k/v bf16, sweep layer_idx 0..31,
        assert output shape / finiteness / cache initialization.
Tier B: real Llama-3.1-8B-Instruct end-to-end generate() with the patch applied.

Usage (in the project venv, on H800):
    BLOCK_SPARSE_NO_AUTOTUNE=1 python reuse_v1/verify_reuse_v1.py --tier a
    python reuse_v1/verify_reuse_v1.py --tier b
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_LABEL = os.environ.get(
    'LLAMA_REUSE_V1_LABEL_PATH',
    os.path.join(_REPO_ROOT, 'ckp', 'Llama-3.1-8B-Instruct', 'full_head.pt'),
)
_DEFAULT_MODEL = os.environ.get(
    'LLAMA_REUSE_V1_MODEL_PATH',
    '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
)


def tier_a(label_path):
    import torch
    from reuse_v1 import get_reuse_v1_prefill
    from reuse_v1.reuse_prefill import reuse_v1_prefill

    device = 'cuda'
    prefill = get_reuse_v1_prefill(label_path=label_path, budget=32,
                                   sink_blocks=1, local_blocks=2, device=device)
    holder = prefill.keywords['holder']
    print(f"[A] label shape={tuple(holder.label.shape)} num_layers={holder.num_layers}")
    assert holder.label.dim() == 2, "label must be 2-D"
    assert bool(holder.label[0].all()), "layer 0 must be all-anchor"

    b, H, Hkv, d, s = 1, 32, 8, 128, 1024
    G = H // Hkv
    torch.manual_seed(0)
    q = torch.randn(b, H, s, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, Hkv, s, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b, Hkv, s, d, dtype=torch.bfloat16, device=device)

    for layer_idx in range(holder.num_layers):
        out = prefill(q, k, v, num_key_value_groups=G, layer_idx=layer_idx)
        assert out.shape == (b, H, s, d), f"bad shape at layer {layer_idx}: {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"non-finite output at layer {layer_idx}"
        if layer_idx == 0:
            assert bool(holder.cache.initialized.all()), "cache not fully initialized after layer 0"
    print(f"[A] swept {holder.num_layers} layers; out shape ok, finite, cache initialized. PASS")


def tier_b(label_path, model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pbs_attn.patch.huggingface import apply_patch_with_prefill
    from reuse_v1 import get_reuse_v1_prefill

    device = 'cuda'
    print(f"[B] loading {model_path} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prefill = get_reuse_v1_prefill(label_path=label_path, budget=32,
                                   sink_blocks=1, local_blocks=2, device=device)
    # No skip_layers: layer 0 MUST be patched so the cache is initialized.
    apply_patch_with_prefill(model, prefill)

    # Long enough (>~800 tokens) to span several 128-blocks and trigger sparsity.
    para = ("The quick brown fox jumps over the lazy dog. "
            "Long-context language models must retrieve relevant facts. ") * 80
    messages = [{"role": "user",
                 "content": para + "\n\nSummarize the repeated theme in one sentence."}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")
    if not hasattr(inputs, 'shape'):  # some versions return a BatchEncoding dict
        inputs = inputs['input_ids']
    inputs = inputs.to(device)
    print(f"[B] prompt tokens = {inputs.shape[-1]}")

    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=32, do_sample=False)
    text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"[B] generated: {text!r}")
    assert text.strip(), "generation is empty"
    print("[B] real-model e2e generate ok, non-empty output. PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['a', 'b'], required=True)
    ap.add_argument('--label-path', default=_DEFAULT_LABEL)
    ap.add_argument('--model-path', default=_DEFAULT_MODEL)
    args = ap.parse_args()

    if args.tier == 'a':
        tier_a(args.label_path)
    else:
        tier_b(args.label_path, args.model_path)


if __name__ == '__main__':
    main()
