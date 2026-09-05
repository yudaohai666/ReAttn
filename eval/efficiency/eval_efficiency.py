import os
import sys
import json
import time
import random
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, HfArgumentParser
from transformers import logging as hf_logging

from tqdm import tqdm

# Silence transformers warnings and reduce distributed verbosity
hf_logging.set_verbosity_error()
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

from pbs_attn.patch.huggingface import (
    apply_patch_with_prefill,
    get_flexprefill_prefill,
    get_minference_prefill,
    get_xattention_prefill,
    get_meanpooling_prefill,
    get_permuted_block_sparse_attn_fwd,
    get_flashattn_prefill,
    get_reuse_v1_prefill,
    get_duo_attention_prefill,
)


@dataclass
class ScriptArguments:
    method: str = field(metadata={"help": "The method to benchmark."})
    len: int = field(default=16 * 1024, metadata={"help": "Length of the input sequence."})
    n_examples: int = field(default=5, metadata={"help": "Number of examples to average over."})
    num_warmup_iter: int = field(default=10, metadata={"help": "Number of warmup iterations."})
    no_save: bool = field(default=False, metadata={"help": "If enabled, do not save the result file."})
    output_dir: str = field(default="results", metadata={"help": "Directory to save results."})
    model_name: str = field(
        default="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct",
        metadata={"help": "HF model id or local path to load (Llama, Qwen2, etc.)."},
    )

@dataclass
class FlexprefillArgs:
    gamma: float = field(default=0.95, metadata={"help": "Gamma for Flexprefill."})
    tau: float = field(default=0.1, metadata={"help": "Tau for Flexprefill."})


@dataclass
class XattnArgs:
    stride: int = field(default=8, metadata={"help": "Stride for XAttention."})
    threshold: float = field(default=0.9, metadata={"help": "Threshold for XAttention."})
    block_size: int = field(default=128, metadata={"help": "Block size for XAttention."})
    keep_sink: bool = field(default=True, metadata={"help": "Keep sink tokens in XAttention."})
    keep_recent: bool = field(default=True, metadata={"help": "Keep recent tokens in XAttention."})


@dataclass
class MeanPoolingArgs:
    block_size: int = field(default=128, metadata={"help": "Block size for MeanPooling."})
    segment_size: int = field(default=1024, metadata={"help": "Segment size for MeanPooling."})
    threshold: float = field(default=0.9, metadata={"help": "Threshold for MeanPooling."})
    force_select_first_block: bool = field(default=True, metadata={"help": "Force select first block for MeanPooling."})
    force_select_current_block: bool = field(default=True, metadata={"help": "Force select current block for MeanPooling."})

@dataclass
class PBSArgs:
    block_size: int = field(default=128, metadata={"help": "Block size for PBS-Attn."})
    segment_size: int = field(default=256, metadata={"help": "Segment size for PBS-Attn."})
    threshold: float = field(default=0.9, metadata={"help": "Threshold for PBS-Attn."})
    force_select_first_block: bool = field(default=True, metadata={"help": "Force select first block for PBS-Attn."})

# --- sparse_reuse ---
@dataclass
class SparseReuseArgs:
    label_path: str = field(metadata={"help": "Path to label .pt file of shape (num_layers, num_kv_heads)."})
    sparse_phase: str = field(default="both", metadata={"help": "'prefill', 'decode', or 'both'."})
    reduce: str = field(default="amax", metadata={"help": "'amax' or 'mean'."})
    sink_blocks: int = field(default=0, metadata={"help": "Always-include first N KV blocks."})
    local_blocks: int = field(default=0, metadata={"help": "Always-include last N KV blocks."})
    topk: int = field(default=16, metadata={"help": "Top-K blocks (16 or 32)."})
    reorder_weights: bool = field(default=False, metadata={"help": "Fold perm into q/k/v/o_proj."})

# --- reuse_v1 (pure-Triton cross-layer per-kv-head block-sparse reuse) ---
@dataclass
class ReuseV1Args:
    label_path: str = field(metadata={"help": "Path to label .pt of shape (num_layers, num_kv_heads); layer 0 all-anchor."})
    budget: int = field(default=32, metadata={"help": "Top-K block budget (16 or 32)."})
    block_size: int = field(default=128, metadata={"help": "Logical block size."})
    segment_size: int = field(default=2048, metadata={"help": "Anchor dense-attn causal segment size."})
    sink_blocks: int = field(default=1, metadata={"help": "Always-include first N KV blocks."})
    local_blocks: int = field(default=2, metadata={"help": "Always-include local diagonal blocks."})
    select_mode: str = field(default="topk", metadata={"help": "Block selection: 'topk' (fixed budget) or 'topp' (nucleus)."})
    top_p: float = field(default=0.9, metadata={"help": "topp: cumulative-mass threshold (only used when select_mode='topp')."})
    min_blocks: int = field(default=8, metadata={"help": "topp: min blocks per (kv_head, q_block), inclusive of sink/diag/local."})
    max_blocks: int = field(default=64, metadata={"help": "topp: max blocks per (kv_head, q_block); sizes kernel headroom/cache width."})
    topk_ratio: float = field(default=None, metadata={"help": "topk: if set, budget = ceil(kv_len//block_size * topk_ratio) + sink_blocks + local_blocks (overrides budget)."})
    last_q_full: bool = field(default=False, metadata={"help": "If True, last query block of sparse kv-heads attends full KV cache."})

# --- duo (DuoAttention: retrieval heads dense + streaming heads sink+local) ---
@dataclass
class DuoArgs:
    attn_load_dir: str = field(metadata={"help": "Dir with full_attention_heads.tsv (+ config.json)."})
    sink_size: int = field(default=128, metadata={"help": "Sink token count for streaming heads."})
    recent_size: int = field(default=256, metadata={"help": "Local/recent window token count for streaming heads."})
    threshold: float = field(default=0.5, metadata={"help": "Score threshold to mark full heads (used when sparsity unset)."})
    sparsity: float = field(default=None, metadata={"help": "Quantile fraction of heads to make streaming; overrides threshold."})
    block_size: int = field(default=128, metadata={"help": "Streaming kernel block size (fixed 128)."})

def build_prefill_fn(method: str, method_args):
    if method == "flexprefill":
        args = method_args or FlexprefillArgs()
        return get_flexprefill_prefill(gamma=args.gamma, tau=args.tau)
    if method == "minference":
        return get_minference_prefill()
    if method == "flashattn":
        return get_flashattn_prefill()
    if method == "xattention":
        args = method_args or XattnArgs()
        return get_xattention_prefill(
            stride=args.stride,
            threshold=args.threshold,
            block_size=args.block_size,
            keep_sink=args.keep_sink,
            keep_recent=args.keep_recent,
        )
    if method == "meanpooling":
        args = method_args or MeanPoolingArgs()
        return get_meanpooling_prefill(
            block_size=args.block_size,
            segment_size=args.segment_size,
            threshold=args.threshold,
            force_select_first_block=args.force_select_first_block,
            force_select_current_block=args.force_select_current_block
        )
    if method == "pbs":
        args = method_args or PBSArgs()
        return get_permuted_block_sparse_attn_fwd(
            block_size=args.block_size,
            segment_size=args.segment_size,
            threshold=args.threshold,
            force_select_first_block=args.force_select_first_block,
            use_triton=True,
        )
    if method == "sparse_reuse":
        return None  # model loading handled separately in main()
    if method == "reuse_v1":
        args = method_args or ReuseV1Args(label_path=None)
        if not args.label_path:
            raise ValueError("reuse_v1 requires --label_path")
        return get_reuse_v1_prefill(
            label_path=args.label_path,
            budget=args.budget,
            block_size=args.block_size,
            segment_size=args.segment_size,
            sink_blocks=args.sink_blocks,
            local_blocks=args.local_blocks,
            select_mode=args.select_mode,
            top_p=args.top_p,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            topk_ratio=args.topk_ratio,
            last_q_full=args.last_q_full,
        )
    if method == "duo":
        args = method_args or DuoArgs(attn_load_dir=None)
        if not args.attn_load_dir:
            raise ValueError("duo requires --attn_load_dir")
        return get_duo_attention_prefill(
            attn_load_dir=args.attn_load_dir,
            sink_size=args.sink_size,
            recent_size=args.recent_size,
            threshold=args.threshold,
            sparsity=args.sparsity,
            block_size=args.block_size,
        )
    raise NotImplementedError(f"Unknown method: {method}")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def init_distributed_if_needed():
    if dist.is_available() and not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group(backend="nccl")


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def warmup_generate(model, tokenizer, length: int, num_iter: int = 10):

    vocab_size = model.get_input_embeddings().weight.size(0)
    input_ids = torch.randint(0, vocab_size, (1, length), device=model.device, dtype=torch.long)

    with torch.no_grad():
        for _ in tqdm(range(num_iter), desc="Warmup", disable=get_rank() != 0):
            # Warm up using generate to exercise the patched prefill path
            _ = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)


def measure_prefill_latency(model, input_ids: torch.Tensor) -> float:
    # Measure E2E prefill using a single-token generate call
    e2e_start = torch.cuda.Event(enable_timing=True)
    e2e_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e2e_start.record()
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)
    e2e_end.record()
    torch.cuda.synchronize()
    return e2e_start.elapsed_time(e2e_end) / 1000.0


def main(script_args, method_args):
    # Seed
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # Distributed setup
    init_distributed_if_needed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Load dataset
    if get_rank() == 0:
        print("Loading LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    long_examples = [d for d in dataset if d['length'] == 'long']

    if len(long_examples) < script_args.n_examples:
        if get_rank() == 0:
            print(f"Warning: Requested {script_args.n_examples} examples, but only {len(long_examples)} long examples found. Using all available.")
        n_examples = len(long_examples)
        sampled_examples = long_examples
    else:
        n_examples = script_args.n_examples
        sampled_examples = random.sample(long_examples, n_examples)

    # Load model/tokenizer
    model_name = script_args.model_name
    if get_rank() == 0:
        print(f"Loading model and tokenizer for {model_name}...")

    if script_args.method == "sparse_reuse":
        # Lazy import: fmha_sm100/cutlass only needed for this method
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from sparse_attn.sparse_reuse import load_model_with_reuse

        args = method_args
        if not args.label_path:
            raise ValueError("sparse_reuse requires --label_path")

        model, tokenizer, cache = load_model_with_reuse(
            model_path=model_name,
            label_path=args.label_path,
            sparse_phase=args.sparse_phase,
            reduce=args.reduce,
            sink_blocks=args.sink_blocks,
            local_blocks=args.local_blocks,
            topk=args.topk,
            dtype=torch.bfloat16,
            device="cuda",
            paged_cache_max_kv_len=script_args.len + 32,
            reorder_weights=args.reorder_weights,
        )
        tokenizer_id = model_name

        if getattr(tokenizer, 'pad_token_id', None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _warmup_fn(model, tokenizer, length, num_iter):
            vocab_size = model.get_input_embeddings().weight.size(0)
            input_ids = torch.randint(0, vocab_size, (1, length), device="cuda", dtype=torch.long)
            with torch.no_grad():
                for _ in tqdm(range(num_iter), desc="Warmup", disable=get_rank() != 0):
                    cache.prepare_for(length + 32)
                    cache.reset()
                    _ = model.generate(input_ids=input_ids, max_new_tokens=1,
                                       do_sample=False, past_key_values=cache, use_cache=True)

        def _measure_fn(model, input_ids):
            e2e_start = torch.cuda.Event(enable_timing=True)
            e2e_end = torch.cuda.Event(enable_timing=True)
            cache.prepare_for(input_ids.shape[1] + 32)
            cache.reset()
            torch.cuda.synchronize()
            e2e_start.record()
            with torch.no_grad():
                _ = model.generate(input_ids=input_ids, max_new_tokens=1,
                                   do_sample=False, past_key_values=cache, use_cache=True)
            e2e_end.record()
            torch.cuda.synchronize()
            return e2e_start.elapsed_time(e2e_end) / 1000.0

    else:
        # Default tokenizer derived from model name
        tokenizer_id = model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if getattr(tokenizer, 'pad_token_id', None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        from_pretrained_kwargs = {"torch_dtype": torch.bfloat16}
        # Enable TP plan if running under torchrun; rely on env-specific support
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            from_pretrained_kwargs["tp_plan"] = "auto"
        else:
            from_pretrained_kwargs["device_map"] = "auto"
        model_config = AutoConfig.from_pretrained(model_name)
        if getattr(model_config, "model_type", None) == "qwen3":
            model_config.rope_parameters = {
                "rope_theta": 1000000,
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
            model_config.max_position_embeddings = 131072
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config, **from_pretrained_kwargs)

        prefill_fn = build_prefill_fn(script_args.method, method_args)
        model = apply_patch_with_prefill(model, prefill_fn)

        _warmup_fn = warmup_generate
        _measure_fn = measure_prefill_latency

    # Warmup
    if get_rank() == 0:
        print("Warming up...")
    if is_distributed():
        dist.barrier()
    _warmup_fn(model, tokenizer, length=script_args.len, num_iter=script_args.num_warmup_iter)

    # Benchmark
    e2e_latencies = []
    # attn_latencies = []  # Not directly measurable here; keep NaN for compatibility
    num_tokens = 0

    if get_rank() == 0:
        print(f"Starting benchmark over {n_examples} examples with method: {script_args.method}...")
    examples_iterable = long_examples if n_examples == len(long_examples) else sampled_examples
    for example in tqdm(examples_iterable, total=n_examples, desc="Benchmark", disable=get_rank() != 0):
        context = example['context']
        inputs = tokenizer(context, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        if script_args.len < input_ids.shape[1]:
            input_ids = input_ids[:, :script_args.len]
        else:
            input_ids = input_ids.repeat(1, script_args.len // input_ids.shape[1] + 1)[:, :script_args.len]

        num_tokens = input_ids.shape[1]

        # Measure prefill latency
        if is_distributed():
            dist.barrier()
        e2e_latency = _measure_fn(model, input_ids)
        if is_distributed():
            tensor_latency = torch.tensor([e2e_latency], device=model.device, dtype=torch.float32)
            dist.all_reduce(tensor_latency, op=dist.ReduceOp.MAX)
            e2e_latency = float(tensor_latency.item())
        e2e_latencies.append(e2e_latency)
        if script_args.method != "sparse_reuse":
            with torch.no_grad():
                _ = model.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False)

    # Aggregate results
    avg_e2e_latency = sum(e2e_latencies) / len(e2e_latencies)
    if get_rank() == 0:
        print(f"\n--- Averaged Results over {n_examples} examples ---")
        print(f"Number of tokens: {num_tokens}")
        print(f"Average E2E Prefill latency: {avg_e2e_latency:.6f} seconds")

    if len(e2e_latencies) > 1:
        e2e_std = torch.tensor(e2e_latencies).std().item()
        if get_rank() == 0:
            print(f"E2E Prefill latency std dev: {e2e_std:.6f} seconds")
    else:
        e2e_std = None

    if get_rank() == 0 and not script_args.no_save:
        if os.path.isabs(script_args.output_dir):
            results_dir = script_args.output_dir
        else:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_args.output_dir)
        os.makedirs(results_dir, exist_ok=True)

        results = {
            "script_args": script_args.__dict__,
            "method_args": method_args.__dict__ if method_args else {},
            "model_name": model_name,
            "tokenizer_name": tokenizer_id,
            "num_tokens": num_tokens,
            "avg_e2e_latency": avg_e2e_latency,
        }
        if e2e_std is not None:
            results["e2e_std"] = e2e_std

        method_args_str = ""
        if method_args:
            items = []
            for key, value in method_args.__dict__.items():
                fname_key = key
                items.append(f"{fname_key}={value}")
            method_args_str = "_" + "_".join(items)

        file_path = os.path.join(results_dir, f"{script_args.method}_{script_args.len}.json")
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {file_path}")


    # Some cluster NCCL setups hang in destroy_process_group after work completes.
    # Keep the default cleanup path, but allow eval-only runs to skip destroy.
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        if os.environ.get("PBS_SKIP_DIST_DESTROY") != "1":
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    method_class_map = {
        "flexprefill": FlexprefillArgs,
        "xattention": XattnArgs,
        "pbs": PBSArgs,
        "meanpooling": MeanPoolingArgs,
        "sparse_reuse": SparseReuseArgs,
        "reuse_v1": ReuseV1Args,
        "duo": DuoArgs,
    }

    method_args = None
    if script_args.method in method_class_map:
        method_class = method_class_map[script_args.method]
        sub_parser = HfArgumentParser(method_class)
        method_args = sub_parser.parse_args_into_dataclasses(args=remaining_args)[0]

    main(script_args, method_args)
