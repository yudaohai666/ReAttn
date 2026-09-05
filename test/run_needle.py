"""Needle-in-a-haystack runner for the pbs-attn sparse-attention project.

Mirrors the QuantPC benchmarks/Needle runner, but instead of a quantized
KV cache it drives the sparse-attention methods that live in this repo:

  * prefill-patch methods (patch the LlamaAttention forward at prefill):
      flashattn, xattention, minference, flexprefill, meanpooling, pbs
  * sparse_reuse: loads its own model + PagedSparseReuseCache (SM100 kernel).

Self-contained under test/; nothing else in the repo is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# Make `import needle` (this dir) and `import pbs_attn / sparse_attn` (repo root)
# resolvable regardless of CWD.
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
for _p in (str(TEST_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from needle import LLMNeedleHaystackTester, ModelProvider  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: E402
from pbs_attn.patch.huggingface import (  # noqa: E402
    apply_patch_with_prefill,
    get_meanpooling_prefill,
    get_minference_prefill,
    get_xattention_prefill,
    get_flexprefill_prefill,
    get_flashattn_prefill,
    get_permuted_block_sparse_attn_fwd,
    get_reuse_v1_prefill,
    get_duo_attention_prefill,
)

DEFAULT_MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
DEFAULT_LABEL = str(REPO_ROOT / "ckp" / "Llama-3.1-8B-Instruct" / "full_head.pt")

# method -> factory that returns a partial(prefill_fn, **kwargs)
PREFILL_FACTORIES = {
    "flashattn": get_flashattn_prefill,
    "minference": get_minference_prefill,
    "xattention": get_xattention_prefill,
    "flexprefill": get_flexprefill_prefill,
    "meanpooling": get_meanpooling_prefill,
    "pbs": get_permuted_block_sparse_attn_fwd,
    "reuse_v1": get_reuse_v1_prefill,
    "duo": get_duo_attention_prefill,
}

DEFAULT_DUO_LABEL_DIR = str(REPO_ROOT / "ckp" / "duo" / "Llama-3.1-8B-Instruct")


class SparseAttnModel(ModelProvider):
    """ModelProvider that applies one of the repo's sparse-attention methods."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        method: str = "flashattn",
        patch_kwargs: Optional[dict] = None,
        attn_impl: str = "flash_attention_2",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 50,
    ):
        patch_kwargs = dict(patch_kwargs or {})
        self.method = method
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Disable thinking mode for Qwen3 models
        self._is_qwen3 = "Qwen3" in model_name
        self._reuse_cache = None

        # Build model config with overrides per model_type
        model_config = AutoConfig.from_pretrained(model_name)
        _model_type = getattr(model_config, "model_type", "")
        if self._is_qwen3:
            model_config.rope_parameters = {
                "rope_theta": 1000000,
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
            model_config.max_position_embeddings = 131072
        elif _model_type == "qwen2":
            # Qwen2.5-1M: neutralize dual_chunk_attention_config so the standard
            # attention forward is used (matching training configuration).
            if hasattr(model_config, "dual_chunk_attention_config"):
                model_config.dual_chunk_attention_config = None

        _needs_config_override = self._is_qwen3 or _model_type == "qwen2"
        if method == "sparse_reuse":
            self._load_sparse_reuse(model_name, patch_kwargs, dtype, model_config if _needs_config_override else None)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                torch_dtype=dtype,
                device_map="cuda",
                attn_implementation=attn_impl,
            )
            self.model.eval()
            if method in PREFILL_FACTORIES:
                prefill_fn = PREFILL_FACTORIES[method](**patch_kwargs)
                self.model = apply_patch_with_prefill(self.model, prefill_fn)
            elif method in ("dense", "none"):
                print("⚠️  No patch applied - dense baseline")
            else:
                raise ValueError(f"unknown method: {method}")

    def _load_sparse_reuse(self, model_name, patch_kwargs, dtype, model_config=None):
        """sparse_reuse replaces the attention impl + drives a paged cache."""
        from sparse_attn.sparse_reuse import load_model_with_reuse

        pk = dict(patch_kwargs)
        label_path = pk.pop("label_path", None)
        if not label_path:
            raise ValueError("sparse_reuse requires 'label_path' in patch_kwargs")
        init_kv_len = pk.pop(
            "paged_cache_max_kv_len", 128 * 1024 + self.max_new_tokens + 64
        )
        print(f"🔧 Loading sparse_reuse model (label={label_path})...")
        model, _tok, cache = load_model_with_reuse(
            model_path=model_name,
            label_path=label_path,
            dtype=dtype,
            device="cuda",
            paged_cache_max_kv_len=int(init_kv_len),
            model_config=model_config,
            **pk,
        )
        model.generation_config.do_sample = False
        model.eval()
        self.model = model
        self._reuse_cache = cache

    def evaluate_model(self, prompt, needle=None) -> str:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            enable_thinking=False if self._is_qwen3 else None,
        ).to(self.model.device)

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # sparse_reuse: size + reset the paged cache and pass it explicitly.
        if self._reuse_cache is not None:
            seq_len = int(input_ids.shape[-1])
            self._reuse_cache.prepare_for(seq_len + self.max_new_tokens + 32)
            self._reuse_cache.reset()
            generate_kwargs["past_key_values"] = self._reuse_cache
            generate_kwargs["use_cache"] = True

        with torch.no_grad():
            output = self.model.generate(**generate_kwargs)
        generated_ids = output[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response

    def generate_prompt(self, context: str, retrieval_question: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI bot that answers questions for a user. "
                    "Keep your response short and direct"
                ),
            },
            {"role": "user", "content": context},
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
            },
        ]

    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        if context_length is not None:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def build_patch_kwargs(args) -> dict:
    """Assemble the per-method patch_kwargs from CLI args."""
    m = args.method
    if m == "xattention":
        return dict(stride=args.stride, threshold=args.threshold, block_size=args.block_size)
    if m == "flashattn":
        return dict(causal=True)
    if m == "sparse_reuse":
        return dict(
            label_path=args.label_path,
            sparse_phase=args.sparse_phase,
            sink_blocks=args.sink_blocks,
            local_blocks=args.local_blocks,
            topk=args.topk,
            dense_tail=args.dense_tail,
        )
    if m == "reuse_v1":
        return dict(
            label_path=args.label_path,
            budget=args.budget,
            block_size=args.block_size,
            segment_size=args.segment_size,
            sink_blocks=args.sink_blocks,
            local_blocks=args.local_blocks,
            causal=True,
            select_mode=args.select_mode,
            top_p=args.top_p,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            last_q_full=args.last_q_full,
        )
    if m == "meanpooling":
        return dict(
            block_size=args.block_size,
            threshold=args.threshold,
            force_select_first_block=True,
            force_select_current_block=True,
        )
    if m == "pbs":
        return dict(
            block_size=args.block_size,
            segment_size=args.segment_size,
            threshold=args.threshold,
            force_select_first_block=True,
        )
    if m == "flexprefill":
        return dict(gamma=args.gamma, tau=args.tau)
    if m == "duo":
        return dict(
            attn_load_dir=args.attn_load_dir,
            sink_size=args.sink_size,
            recent_size=args.recent_size,
            sparsity=args.sparsity,
            block_size=args.block_size,
            causal=True,
        )
    if m == "minference":
        return dict(vertical_size=args.vertical_size, slash_size=args.slash_size, adaptive_budget=None)
    return {}


def run_experiment(args) -> dict:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = SparseAttnModel(
        model_name=args.model,
        method=args.method,
        patch_kwargs=build_patch_kwargs(args),
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
    )
    tester = LLMNeedleHaystackTester(model)
    results: dict = {}

    # Resume: load previously completed (length, depth) results so we skip them.
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for key, value in prev.items():
                length_str, _, depth_str = key.partition("-")
                results[(int(length_str), int(depth_str))] = value
            print(f"[resume] loaded {len(results)} completed cases from {output_path}")
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            print(f"[resume] ignoring unreadable checkpoint {output_path}: {exc}")

    for length in parse_int_list(args.lengths):
        for depth in parse_int_list(args.depths):
            if (length, depth) in results:
                print((length, depth), "[skip] already done")
                continue
            results[(length, depth)] = tester.evaluate(length, depth)
            print((length, depth), results[(length, depth)])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({f"{k[0]}-{k[1]}": v for k, v in results.items()}, f, indent=4)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-a-haystack with sparse attention.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--method",
        default="flashattn",
        help="flashattn | xattention | sparse_reuse | reuse_v1 | pbs | meanpooling | minference | flexprefill | dense",
    )
    parser.add_argument("--lengths", default="8192,16384,32768,65536,131072", help="comma-separated context lengths")
    parser.add_argument("--depths", default="0,25,50,75,100", help="comma-separated needle depths")
    parser.add_argument("--output", default="results/niah/results.json")
    parser.add_argument("--attn-impl", default="flash_attention_2", help="decode-path attn impl (flash_attention_2/sdpa/eager)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    # xattention / shared block params
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--segment-size", type=int, default=256)
    # flexprefill
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.1)
    # minference
    parser.add_argument("--vertical-size", type=int, default=1000)
    parser.add_argument("--slash-size", type=int, default=6096)
    # sparse_reuse
    parser.add_argument("--label-path", default=DEFAULT_LABEL)
    parser.add_argument("--sparse-phase", default="prefill", help="prefill | decode | both")
    parser.add_argument("--sink-blocks", type=int, default=1)
    parser.add_argument("--local-blocks", type=int, default=2)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--dense-tail", type=int, default=0,
                        help="last N sparse-head query rows attend densely (prefill); 0 disables")
    # reuse_v1
    parser.add_argument("--budget", type=int, default=32, help="reuse_v1 top-k block budget (16 or 32)")
    parser.add_argument("--select-mode", default="topk", choices=["topk", "topp"],
                        help="reuse_v1 block selection: 'topk' (fixed budget) or 'topp' (nucleus)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="reuse_v1 topp nucleus coverage (select_mode=topp)")
    parser.add_argument("--min-blocks", type=int, default=8,
                        help="reuse_v1 topp min selected blocks per q-block (incl. sink+local)")
    parser.add_argument("--max-blocks", type=int, default=64,
                        help="reuse_v1 topp max selected blocks per q-block (incl. sink+local; "
                             "None -> kernel headroom max_sel)")
    parser.add_argument("--last-q-full", action="store_true",
                        help="reuse_v1: last query block of sparse kv-heads attends densely "
                             "to the full KV cache (better retrieval recall)")
    # duo (DuoAttention: retrieval heads dense + streaming heads sink+local)
    parser.add_argument("--attn-load-dir", default=DEFAULT_DUO_LABEL_DIR,
                        help="duo: dir with full_attention_heads.tsv (+ config.json)")
    parser.add_argument("--sink-size", type=int, default=128, help="duo: streaming-head sink token count")
    parser.add_argument("--recent-size", type=int, default=256, help="duo: streaming-head local window token count")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="duo: quantile fraction of kv-heads made streaming (0..1)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run_experiment(args)


if __name__ == "__main__":
    main()
