#!/usr/bin/env bash
# Train the reuse_v1 per-(layer, kv-head) anchor/sparse gate with Stochastic
# Gates (--reg_mode stg) + adaptive-Lagrangian density constraint.
#
# STG parameterization (official runopti/stg code convention):
#   anchor_heads = mu_code  (logit; init ~0.0 -> gate starts at clip(0.5+sigma*eps, 0,1))
#   z = clip(mu_code + 0.5 + sigma * eps, 0, 1),  eps ~ N(0,1) per step
#   density = Phi((mu_code + 0.5) / sigma)  (expected anchor fraction, Gaussian CDF)
#   Lagrangian: single adaptive lambda0 (multiplicative EMA, same as kuma)
#   Export: deterministic top-k on mu_code (monotone with density) -> same as ddp
#
# Fixed internals (NOT exposed as args):
#   lagrange_alpha=0.9, lambda clamp [1e-12, 20]
#   (from reuse_v1/utils.py defaults / train_reuse.py constants)
#
# The reuse_v1 inference hyperparameters (budget=32, block_size=128,
# segment_size=2048, sink_blocks=1, local_blocks=2) are LOCKED inside
# reuse_v1/train_reuse.py and MUST match inference, so they are NOT args.
# Block selection here is TOPP (nucleus: top_p=0.9, min_blocks=8, max_blocks=64).
#
# Usage:
#   bash scripts/train_reuse_stg.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [target_sparsity] [stg_sigma] [lagrange_lr] [lambda_init_value] [no_ac]
#
#   sp_size          Ulysses SP group size. Default 8 (Llama-3.1-8B 32q/8kv,
#                    128k fits ~12.8GB/rank). sp_size=1 = pure FSDP2 DP (OOMs at 128k).
#   target_sparsity  Target fraction of (layer>=1) kv-heads that end up SPARSE.
#                    Default 0.7695 (matches ddp/kuma reference sparsity).
#   stg_sigma        Gaussian noise scale. Default 0.5 (official STG).
#                    Lower (e.g. 0.3) -> sharper gates, less exploration.
#   lagrange_lr      Adaptive lambda learning rate (exp multiplicative). Default 0.001.
#   lambda_init_value Initial lambda0 value. Default 1.0.
#   no_ac            1 = disable activation checkpointing (default 1, safe with sp8).
#                    0 = enable AC (reduces activation memory, slower per step).
#   top_p            Nucleus coverage for topp block selection. Default 0.9.
#                    Lower (e.g. 0.7) -> larger sparse/dense gap -> stronger gate gradient.
#                    MUST match inference top_p.
set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
# Long-sequence prefill fragments the allocator; expandable segments avoids OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve repo root from this script's location so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

model_name=${1}
ctx_len_min=${2}
ctx_len_max=${3}
lr=${4}
num_passkey=${5}
sp_size=${6:-8}
target_sparsity=${7:-0.7695}
stg_sigma=${8:-0.5}
lagrange_lr=${9:-0.001}
lambda_init_value=${10:-1.0}
no_ac=${11:-1}   # 1 = 关闭 AC（默认关闭，仅 sp_size>1 时安全）
top_p=${12:-0.9}

setting="stg-sp=${target_sparsity}-sigma=${stg_sigma}-lr=${lr}-llr=${lagrange_lr}-linit=${lambda_init_value}-topp=${top_p}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
exp_name="reuse_v1/${model_name}/${setting}"

torchrun --nnodes 1 --nproc_per_node 8 \
    reuse_v1/train_reuse.py \
    --model_name "${model_name}" \
    --batch_size 1 \
    --max_length "${ctx_len_max}" \
    --dataset_name "datasets/PaulGrahamEssays.jsonl" \
    --dataset_format "multiple_passkey" \
    --num_steps 2000 \
    --lr "${lr}" \
    --reg_mode stg \
    --target_sparsity "${target_sparsity}" \
    --stg_sigma "${stg_sigma}" \
    --lagrange_lr "${lagrange_lr}" \
    --lambda_init_value "${lambda_init_value}" \
    --initial_value 0.0 \
    --select_mode topp \
    --top_p "${top_p}" \
    --min_blocks 8 \
    --max_blocks 64 \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min "${ctx_len_min}" \
    --context_length_max "${ctx_len_max}" \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkeys "${num_passkey}" \
    --save_steps 50 \
    --two_pass \
    ${no_ac:+--no_ac} \
    --sp_size "${sp_size}" \
    --output_dir "attn_patterns/${exp_name}"
