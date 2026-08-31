#!/usr/bin/env bash
# Train the reuse_v1 per-(layer, kv-head) anchor/sparse gate with HC + Lagrangian
# (--reg_mode hcl).
#
# HCL = Hard Concrete (Louizos et al. ICLR 2018) gate + Lagrangian density
# constraint. The gate sampling is identical to --reg_mode hc:
#
#   u  ~ Uniform(0,1)  [frozen per step]
#   t  = sigmoid((logit(u) + log_alpha) / beta)   [beta=2/3 fixed]
#   z  = clamp( t * (zeta-gamma) + gamma, 0, 1 )  [gamma=-0.1, zeta=1.1]
#
# The KEY difference from --reg_mode hc is the sparsity regularizer:
#   hc  uses L0 penalty: reg_weight * sum P(z != 0) = sum sigmoid(alpha + 1.599)
#   hcl uses Lagrangian: lambda * relu( mean(sigmoid(alpha)) - desired_density )
#
# Why hcl avoids the export sparsity gap:
#   density  = mean(sigmoid(alpha)) = mean(P(z > 0.5))
#   export threshold: alpha > 0  <=>  sigmoid(alpha) > 0.5
#   At convergence: mean(sigmoid(alpha)) ≈ desired_density
#     and fraction(alpha > 0) ≈ desired_density  (both use sigmoid(alpha))
#   => export sparsity ≈ target_sparsity (no gap, unlike hc L0 or hln).
#
# Lagrangian density constraint (NOT a fixed penalty):
#   loss = distill_loss + lambda * relu(mean(sigmoid(alpha)) - desired_density)
#   lambda updated multiplicatively each step.
#   desired_density = 1.0 - target_sparsity
#
# Usage:
#   bash scripts/train_reuse_hcl.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [target_sparsity] [initial_value]
#
#   sp_size          Ulysses SP group size. Default 8.
#   target_sparsity  Fraction of heads to make sparse. Default 0.79.
#   initial_value    Initial log_alpha value. Default 1.0 (sigmoid(1.0)=0.73 warm-start).
set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

model_name=${1}
ctx_len_min=${2}
ctx_len_max=${3}
lr=${4}
num_passkey=${5}
sp_size=${6:-8}
target_sparsity=${7:-0.79}
initial_value=${8:-1.0}

setting="hcl-sp=${target_sparsity}-init=${initial_value}-lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
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
    --reg_mode hcl \
    --lagrange_lr 0.001 \
    --target_sparsity "${target_sparsity}" \
    --initial_value "${initial_value}" \
    --select_mode topp \
    --top_p 0.9 \
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
    --no_ac \
    --sp_size "${sp_size}" \
    --output_dir "attn_patterns/${exp_name}"
