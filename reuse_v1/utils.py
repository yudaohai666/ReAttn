import argparse
import transformers
import torch
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def parse_args():
    parser = argparse.ArgumentParser(description="kv_reduction")

    parser.add_argument(
        "--model_name", type=str, default="/home/guangxuanx/models/LLaMA-2-7B-32K"
    )
    parser.add_argument("--config_name", type=str, default=None)

    # train params
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="/home/guangxuanx/datasets/Long-Data-Collections/pretrain/pile_sub.jsonl.zst",
    )
    parser.add_argument("--dataset_format", type=str, default="multiple_passkey")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--context_length_min", type=int, default=1024)
    parser.add_argument("--context_length_max", type=int, default=4096)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=20)
    parser.add_argument("--depth_ratio_num_intervals", type=int, default=10)
    parser.add_argument("--num_passkeys", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sink_size", type=int, default=64)
    parser.add_argument("--recent_size", type=int, default=256)
    parser.add_argument("--deploy_sink_size", type=int, default=None)
    parser.add_argument("--deploy_recent_size", type=int, default=None)
    parser.add_argument("--reg_weight", type=float, default=0.05)
    parser.add_argument(
        "--reg_mode",
        type=str,
        default="l1",
        choices=["l1", "kuma", "ddp", "stg", "hc", "hln", "hcl"],
        help="reuse_v1 training regularizer / gate. 'l1' (default): deterministic "
        "clamped anchor_heads gate + L1 sparsity (byte-identical to the original "
        "path; --reg_weight applies). 'kuma': HardKuma stochastic gate (trainable "
        "per-(layer,kv-head) shape params a,b; z reparameterized inside each layer "
        "forward from frozen per-step noise) + an adaptive-lambda Lagrangian "
        "density constraint driving E[anchor fraction] toward --desired_density "
        "(--reg_weight is IGNORED; lambda self-adapts). 'ddp': deterministic "
        "differentiable pruning -- anchor_heads IS the learnable logit z_loga, "
        "blend gate = straight-through clamp(z_loga,0,1); a 3-term LEARNABLE-lambda "
        "Lagrangian on the annealed soft-saturation score drives the expected "
        "SPARSE fraction toward --target_sparsity (--reg_weight IGNORED). Export "
        "uses a deterministic top-k head mask hitting the exact target. Layer 0 "
        "stays all-anchor. 'stg': Stochastic Gates -- anchor_heads IS mu_code, "
        "z = clip(mu+0.5+sigma*eps, 0,1), eps~N(0,1); adaptive single-lambda "
        "Lagrangian drives mean(Phi(mu/sigma)) toward 1-target_sparsity; "
        "export = global top-k on mu_code. Layer 0 stays all-anchor. "
        "'hc': Hard Concrete (Louizos et al. ICLR 2018) -- anchor_heads IS "
        "log_alpha, z = clamp(sigmoid((logit(u)+log_alpha)/beta)*(zeta-gamma)+gamma,0,1), "
        "u~Uniform(0,1) frozen per step; single-parameter per head; adaptive "
        "single-lambda Lagrangian drives mean(P(z>0)) toward 1-target_sparsity; "
        "export = global top-k on log_alpha. Layer 0 stays all-anchor. "
        "'hcl': Hard Concrete + Lagrangian -- same HC gate as 'hc' (anchor_heads IS "
        "log_alpha, z = clamp(sigmoid((logit(u)+log_alpha)/beta)*(zeta-gamma)+gamma,0,1)); "
        "density = mean(sigmoid(log_alpha)) = mean(P(z>0.5)), aligned with export "
        "threshold log_alpha>0; adaptive single-lambda Lagrangian drives density "
        "toward 1-target_sparsity; export = log_alpha>0 (or top-k). "
        "Layer 0 stays all-anchor. "
        "'hln': Hard Logistic-Normal -- anchor_heads IS mu; "
        "z = clamp(sigmoid(mu+sigma*eps)*(zeta-gamma)+gamma, 0,1), eps~N(0,1) "
        "frozen per step; Normal->sigmoid->stretch->clip structure (like HardKuma) "
        "gives nonzero mass at 0 and 1; adaptive single-lambda Lagrangian drives "
        "mean(Phi(mu/sigma)) toward 1-target_sparsity; sigma is --hln_sigma "
        "(global temperature, not per-head); export = global top-k on mu. "
        "Layer 0 stays all-anchor.",
    )
    parser.add_argument(
        "--desired_density",
        type=float,
        default=0.08333,
        help="--reg_mode kuma only: target expected anchor (dense-head) fraction "
        "over ALL layers (layer 0 included, matching the reference train_kuma; "
        "layer 0 is frozen all-anchor so it contributes a ~0.917 constant to the "
        "mean). The Lagrangian constraint pushes density down toward this value; "
        "lower -> sparser label. Default 0.08333 matches the reference "
        "train_kuma_multi_passkey.sh (very aggressive; raise for a denser label).",
    )
    parser.add_argument(
        "--lagrange_lr",
        type=float,
        default=0.001,
        help="--reg_mode kuma/stg: multiplicative step for the adaptive Lagrange "
        "multiplier update lambda *= exp(lagrange_lr * constraint_violation). "
        "Default 0.001 matches the reference train_kuma_multi_passkey.sh.",
    )
    parser.add_argument(
        "--lamda_init_value",
        type=float,
        default=2.0,
        help="--reg_mode kuma only: initial Lagrange multiplier lambda0 "
        "(clamped to [1e-12, 20.0] each step). Default 2.0 matches the reference "
        "train_kuma_multi_passkey.sh.",
    )
    parser.add_argument("--initial_value", type=float, default=1.0)
    # --- reg_mode ddp only ------------------------------------------------
    parser.add_argument(
        "--target_sparsity",
        type=float,
        default=0.7695,
        help="--reg_mode ddp only: target fraction of (layer>=1) kv-heads that "
        "are SPARSE (zeroed). The learnable-lambda Lagrangian drives the annealed "
        "expected sparsity 1 - score.mean() toward this; export zeroes exactly "
        "round(target_sparsity * L * H) lowest-scoring heads. NOTE: the DDP "
        "reference (train.sh) prunes only 0.20; 0.7695 here is a reuse_v1 goal "
        "choice (~77% sparse, matching the kuma density=0.23 run). Lower for a "
        "denser label.",
    )
    parser.add_argument(
        "--lambda_1_lr",
        type=float,
        default=2e-2,
        help="--reg_mode ddp only: dual-ascent lr for the linear multiplier "
        "lambda_1 on (expected_sparsity - target). Default 2e-2 (DDP reference).",
    )
    parser.add_argument(
        "--lambda_2_lr",
        type=float,
        default=4e-1,
        help="--reg_mode ddp only: dual-ascent lr for the quadratic multiplier "
        "lambda_2 on (expected_sparsity - target)^2. Default 4e-1 (DDP reference).",
    )
    parser.add_argument(
        "--lambda_3_lr",
        type=float,
        default=None,
        help="--reg_mode ddp only: dual-ascent lr for the binary-entropy "
        "multiplier lambda_3 on mean((1-score)*score). None -> falls back to "
        "--lambda_1_lr. Ignored if --no_binary_loss.",
    )
    parser.add_argument(
        "--no_binary_loss",
        action="store_true",
        help="--reg_mode ddp only: drop the lambda_3 binary-entropy term "
        "((1-score)*score, pushes score toward hard 0/1).",
    )
    parser.add_argument(
        "--uniform_sparsity",
        action="store_true",
        help="--reg_mode ddp only: export with a per-LAYER top-k (each layer>=1 "
        "zeroes the same round(target_sparsity*H) count) instead of one global "
        "top-k over all (layer>=1, head) slots.",
    )
    parser.add_argument(
        "--z_loga_clamp_min",
        type=float,
        default=-0.1,
        help="--reg_mode ddp only: per-step lower floor for the raw logit "
        "z_loga (anchor_heads). Upper clamp is fixed at 1.1 (LIMIT_B) to prevent "
        "soft_saturation_score from saturating. Default -0.1 (DDP reference).",
    )
    parser.add_argument(
        "--lambda_init_value",
        type=float,
        default=1.0,
        help="--reg_mode ddp/stg only: initial value for learnable Lagrange multipliers "
        "lambda_1, lambda_2, lambda_3 (ddp) or lambda0 (stg). Non-zero init ensures "
        "sparsity pressure from step 1 (cold-start). Default 1.0.",
    )
    parser.add_argument(
        "--stg_sigma",
        type=float,
        default=0.5,
        help="--reg_mode stg only: noise scale sigma for the Gaussian stochastic gate. "
        "z = clip(mu + 0.5 + sigma * eps, 0, 1). "
        "Default 0.5 (official STG). Smaller values (e.g. 0.3) concentrate the "
        "distribution and speed up polarization.",
    )
    parser.add_argument(
        "--hln_sigma",
        type=float,
        default=0.5,
        help="--reg_mode hln only: noise scale sigma (global temperature) for the "
        "Hard Logistic-Normal gate. z = clamp(sigmoid(mu+sigma*eps)*(zeta-gamma)+gamma,0,1). "
        "density = Phi(mu/sigma); sigma is NOT per-head to avoid hijacking the Lagrangian. "
        "Default 0.5. Smaller values sharpen the gate; minimum enforced at HLN_SIGMA_MIN.",
    )
    parser.add_argument(
        "--anneal_schedule",
        type=str,
        default="sqrt",
        choices=["sqrt", "linear", "quad"],
        help="--reg_mode ddp only: how the soft-saturation mean anneals from "
        "0.5 down to --anneal_mean_min over training. Default sqrt.",
    )
    parser.add_argument(
        "--anneal_mean_min",
        type=float,
        default=0.1,
        help="--reg_mode ddp only: final (min) soft-saturation mean; smaller -> "
        "sharper gate. Default 0.1 (DDP reference).",
    )
    parser.add_argument(
        "--anneal_warmup_ratio",
        type=float,
        default=0.0,
        help="--reg_mode ddp only: hold the soft-saturation mean at 0.5 for this "
        "initial fraction of steps before annealing. Default 0.0.",
    )
    parser.add_argument(
        "--z_loga_init_std",
        type=float,
        default=1e-2,
        help="--reg_mode ddp only: std of the Gaussian noise added to the logit "
        "init (mean=--initial_value) for layers >=1. Default 1e-2 (DDP reference).",
    )
    parser.add_argument(
        "--select_mode",
        type=str,
        default="topk",
        choices=["topk", "topp"],
        help="reuse_v1 training: student sparse-branch block selection. 'topk' "
        "(default) keeps a fixed --budget blocks (byte-identical to the original "
        "path). 'topp' = nucleus: keep the fewest top-scored blocks whose "
        "cumulative mean-mass >= --top_p, count clamped to [--min_blocks, "
        "--max_blocks] (--budget then only sizes the cache-fallback width). MUST "
        "match the reuse_v1 INFERENCE selection or the exported head label will "
        "not transfer.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="--select_mode topp only: cumulative mean-mass threshold (nucleus).",
    )
    parser.add_argument(
        "--min_blocks",
        type=int,
        default=8,
        help="--select_mode topp only: lower bound on the FINAL per-kv-head "
        "block count (includes sink + diagonal + local).",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=64,
        help="--select_mode topp only: upper bound on the FINAL per-kv-head "
        "block count; also sizes the selection cache width (max_sel).",
    )
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--enable_pp", action="store_true")
    parser.add_argument("--enable_tp", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--min_needle_depth_ratio", type=float, default=0)
    parser.add_argument("--max_needle_depth_ratio", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sp_size",
        type=int,
        default=1,
        help="reuse_v1 --two_pass only: Ulysses sequence-parallel group size. "
        "1 (default) = pure FSDP2 data parallelism, each rank runs the FULL "
        "sequence (byte-identical to the old path). >1 shards one sequence "
        "across sp_size ranks (each holds seq_len/sp_size tokens); attention "
        "does an internal all-to-all so every rank sees the WHOLE sequence for "
        "its head subset -> global top-k block selection stays EXACT. Requires "
        "world_size %% sp_size == 0 and num_attention_heads / num_key_value_heads "
        "both divisible by sp_size. dp_size = world_size // sp_size runs distinct "
        "samples. Lets 128k fit at ~12.8GB/rank without CPU offload.",
    )
    parser.add_argument(
        "--two_pass",
        action="store_true",
        help="reuse_v1 training: split the forward into a no_grad fill pass "
        "(populates cross-layer top-k selections) + a grad use pass that reads "
        "the frozen selections. Enables activation checkpointing for long "
        "sequences (gradient-exact vs the single-pass forward).",
    )
    parser.add_argument(
        "--ac_group_size",
        type=int,
        default=1,
        help="reuse_v1 --two_pass only: group this many consecutive "
        "LlamaDecoderLayers into ONE activation-checkpoint segment. Default 1 "
        "(per-layer AC). NOTE: values >1 are counterproductive for reuse_v1 -- "
        "the anchor/dense path allocates a large fp32 block-score scratch (~12.6GB "
        "at 128k) that is recomputed in backward, and a multi-layer segment holds "
        "one such scratch PER layer simultaneously, which OOMs. Keep at 1.",
    )
    parser.add_argument(
        "--no_ac",
        action="store_true",
        help="reuse_v1 --two_pass only: disable activation checkpointing even "
        "when --two_pass is set. Saves ~20-25%% per-step time at the cost of "
        "higher activation memory. Safe under sp_size>1 (each rank only holds "
        "seq/sp_size tokens, so activation memory is proportionally reduced). "
        "Do NOT use with sp_size=1 at 128k (will OOM).",
    )
    parser.add_argument(
        "--hc_force_layer0",
        action="store_true",
        help="--reg_mode hc/hcl only: revert to the old design where layer 0 is "
        "forced all-anchor (gate frozen at 1, no streaming fallback). When set, "
        "hc behaves like the pre-streaming-fallback mode: layer 0 always dense, "
        "L0 excluded from the L0 penalty, and label[0] forced True at export. "
        "Default off (new streaming-fallback design).",
    )
    parser.add_argument("--rope_theta", type=float, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--streaming_attn_implementation", type=str, default="blocksparse"
    )

    parser.add_argument(
        "--supervision",
        type=str,
        default="distill",
        choices=["classify", "distill"],
    )

    # Eval params
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--task", type=str, default="default")
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--passkey_length", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--generation_length", type=int, default=256)
    parser.add_argument("--stride_length", type=int, default=256)
    parser.add_argument("--prefilling_chunk_size", type=int, default=4096)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.device = parse_device(args.device)
    return args


def parse_device(device: str):
    if "," in device:
        return [int(d) for d in device.split(",")]
    elif device in ["auto", "cpu"]:
        return device
    return f"cuda:{device}"


def get_model(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    if hasattr(model.config, "sliding_window") and model.config.sliding_window is None:
        model.config.sliding_window = model.config.max_position_embeddings

    return model


from transformers import (
    PretrainedConfig,
)
from typing import Sequence
from functools import partial
import re

# NOTE: ``tensor_parallel`` is an OPTIONAL heavy dependency used only by the
# legacy TP helpers (``get_mistral_config`` / the ``enable_tp`` branch of
# ``to_device``). The FSDP2 training entrypoint (train_reuse.py) does not use
# it, so its imports are deferred into those functions to keep this module
# importable without tensor_parallel installed.


def get_mistral_config(
    model_config: PretrainedConfig, devices: Sequence[torch.device]
) -> "Config":
    from tensor_parallel.config import Config
    from tensor_parallel.communications import CollectiveOperation
    from tensor_parallel.aux_actions import (
        gather_kv,
        select_kv_for_rank,
        split_inner_dim,
        split_num_heads,
    )
    from tensor_parallel.state_actions import Split, SplitInChunks

    assert (
        model_config.model_type == "mistral"
    ), f"Trying to pass {model_config.model_type} as mistral config"

    world_size = len(devices)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    num_kv = model_config.num_key_value_heads
    q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    config = Config(
        state_rules={
            # MistralAttention
            r".*self_attn\.q_proj\.weight$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=q_per_kv * head_dim
            ),
            r".*self_attn\.k_proj\.weight$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=head_dim
            ),
            r".*self_attn\.v_proj\.weight$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=head_dim
            ),
            r".*self_attn\.o_proj\.weight$": SplitInChunks(
                world_size=world_size, dim=1, chunk_size=q_per_kv * head_dim
            ),
            # MistralMLP
            r".*mlp\.gate_proj\.weight$": Split(world_size=world_size, dim=0),
            r".*mlp\.down_proj\.weight$": Split(world_size=world_size, dim=1),
            r".*mlp\.up_proj\.weight$": Split(world_size=world_size, dim=0),
            # MistralModel
            r".*embed_tokens.weight$": Split(world_size=world_size, dim=1),
            r".*lm_head\.weight$": Split(world_size=world_size, dim=0),
        },
        input_rules={
            r".*self_attn$": {"past_key_value": select_kv_for_rank},
        },
        output_rules={
            r".*self_attn$": {0: "sum", 2: gather_kv_across_ranks},
            r".*mlp$": {0: "sum"},
            r".*embed_tokens$": {0: "gather -1"},
            r".*lm_head$": {0: "gather -1"},
        },
        attr_rules={
            r".*self_attn$": {
                "hidden_size": partial(
                    split_inner_dim, num_heads=num_kv, world_size=world_size
                ),
                "num_heads": lambda n, rank: q_per_kv
                * split_num_heads(n // q_per_kv, rank=rank, world_size=world_size),
            }
        },
    )

    config.attr_rules[re.compile(".*self_attn$")]["num_key_value_heads"] = partial(
        split_num_heads, world_size=world_size
    )

    return config


def to_device(
    model,
    device,
    enable_tp=False,
    enable_pp=False,
    reverse_device_map=True,
    even_split_layers=True,
):
    if enable_tp and isinstance(device, list):
        if len(device) == 1:
            return model.to(f"cuda:{device[0]}")
        import tensor_parallel as tp
        from tensor_parallel.pretrained_model import (
            find_predefined_tensor_parallel_config,
        )
        from tensor_parallel.autoconfig import get_default_config
        from tensor_parallel.state_actions import Split

        device_ids = [f"cuda:{idx}" for idx in device]
        world_size = len(device_ids)
        tensor_parallel_config = find_predefined_tensor_parallel_config(
            model.config, device_ids
        )
        if tensor_parallel_config is None:
            if model.config.model_type == "mistral":
                tensor_parallel_config = get_mistral_config(model.config, device_ids)
            else:
                tensor_parallel_config = get_default_config(model, device_ids)
        tensor_parallel_config.state_rules[re.compile(r".*full_attention_heads$")] = (
            Split(world_size=world_size, dim=0)
        )
        return tp.tensor_parallel(
            model,
            device_ids,
            tensor_parallel_config=tensor_parallel_config,
            sharded=True,
        )
    elif enable_pp and isinstance(device, list):
        no_split_module_classes = [
            "MistralDecoderLayer",
            "LlamaDecoderLayer",
        ]
        max_memory = {
            device_id: torch.cuda.get_device_properties(device_id).total_memory
            for device_id in device
        }
        print("Max Memory:", max_memory)
        max_memory = get_balanced_memory(
            model,
            max_memory,
            no_split_module_classes=no_split_module_classes,
        )
        device_map = infer_auto_device_map(
            model, max_memory, no_split_module_classes=no_split_module_classes
        )
        modules = list(device_map.keys())
        num_devices = len(device)
        device_map = {}
        current_device_idx = 0

        if even_split_layers:
            num_layers_per_device = model.config.num_hidden_layers / num_devices
            current_layer_idx = 0
            current_other_idx = 0
            for idx, module in enumerate(modules):
                if "layer" in module:
                    device_map[module] = device[current_device_idx]
                    current_layer_idx += 1
                    if current_layer_idx >= num_layers_per_device:
                        current_device_idx += 1
                        current_layer_idx = 0
                elif "lm_head" in module:
                    device_map[module] = device[-1]
                elif "norm" in module:
                    device_map[module] = device[-1]
                elif "embed" in module:
                    device_map[module] = device[0]
                else:
                    device_map[module] = device[current_other_idx]
                    current_other_idx += 1
                    continue
        else:
            num_modules_per_device = len(modules) / num_devices
            for idx, module in enumerate(modules):
                device_map[module] = device[current_device_idx]
                if (idx + 1) >= num_modules_per_device * (current_device_idx + 1):
                    current_device_idx += 1

        if reverse_device_map:
            device_map = {k: num_devices - v - 1 for k, v in device_map.items()}
        print("Device Map:", device_map)
        dispatch_model(model, device_map)
        return model
    else:
        return model.to(device)


def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def full_attention_heads_to_list(full_attention_heads):
    num_pruned_layers = len(full_attention_heads)
    num_heads = full_attention_heads[0].shape[0]
    for idx in range(num_pruned_layers):
        full_attention_heads[idx] = (
            full_attention_heads[idx].detach().float().cpu().tolist()
        )
    return full_attention_heads


def visualize_pruned_attention_heads(full_attention_heads):
    img = np.array(full_attention_heads)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="coolwarm", interpolation="nearest")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.colorbar(fraction=0.046, pad=0.04)
    # scale the color to 0-1
    plt.clim(0, 1)
    plt.tight_layout()
    plt.title("Ratio of Full Attention Computations")
    return fig


def load_attn_pattern(attn_load_dir):
    full_attention_heads = np.loadtxt(
        os.path.join(attn_load_dir, "full_attention_heads.tsv"),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    config = json.load(open(os.path.join(attn_load_dir, "config.json")))
    sink_size = config["sink_size"]
    recent_size = config["recent_size"]
    return full_attention_heads, sink_size, recent_size


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sparsify_attention_heads(full_attention_heads, threshold=None, sparsity=None):
    # add a very small random noise to full_attention_heads to break ties
    full_attention_heads += np.random.uniform(0, 1e-6, full_attention_heads.shape)

    if sparsity is not None:
        # ignore the threshold and use the sparsity
        # set the sparsity small values to 0 and others to 1
        threshold = np.quantile(full_attention_heads, sparsity)
    else:
        assert threshold is not None, "Either threshold or sparsity must be provided"

    if sparsity >= 1:
        # all heads are pruned
        threshold = 2
    if sparsity <= 0:
        # no heads are pruned
        threshold = -1

    full_attention_heads = (full_attention_heads >= threshold).astype(float)
    sparsity = 1 - np.mean(full_attention_heads)
    return full_attention_heads, sparsity


def save_full_attention_heads(full_attention_heads, output_filename):
    np.savetxt(
        output_filename,
        np.array(full_attention_heads),
        delimiter="\t",
    )
