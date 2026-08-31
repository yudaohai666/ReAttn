#!/usr/bin/env bash
# Train the reuse_v1 per-(layer, kv-head) anchor/sparse gate with HardLogistic
# (--reg_mode hln), HC with learnable global temperature s + augmented Lagrangian.
#
# HardLogistic(mu, s) = Hard Concrete with learnable s replacing fixed beta:
#   u  ~ Uniform(0,1)  [frozen per step]
#   z  = clamp( sigmoid((logit(u) + mu) / s) * 1.2 - 0.1, 0, 1 )
#   density(mu) = sigmoid(mu)   [s-free, s cancels at z>0.5 threshold]
#
# Lagrangian density constraint (NOT a fixed penalty):
#   loss = distill_loss + lambda * relu(mean sigmoid(mu) - desired_density)
#   lambda updated multiplicatively each step to enforce the constraint.
#   desired_density = 1.0 - target_sparsity
#
# Advantages over HC and HardKuma:
#   * s-free density: Lagrangian drives mu only; s cannot hijack sparsity.
#   * No dead zone: d/dmu sigmoid(mu)|_{mu=0} = 0.25 (max gradient, unlike Kuma).
#   * Export threshold mu>0 aligns exactly with sigmoid(mu)>0.5.
#   * s learnable: gate sharpness adapts (warm exploration -> convergence hardening).
#
# Usage:
#   bash scripts/train_reuse_hln.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [target_sparsity] [initial_value] [hln_sigma]
#
#   sp_size          Ulysses SP group size. Default 8.
#   target_sparsity  Fraction of heads to make sparse. Default 0.79.
#   initial_value    Initial mu value. Default 1.0 (sigmoid(1.0)=0.73 warm-start).
#   hln_sigma        Initial s temperature. Default 0.5.
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
hln_sigma=${9:-0.5}

setting="hln-sp=${target_sparsity}-init=${initial_value}-s=${hln_sigma}-lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
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
    --reg_mode hln \
    --target_sparsity "${target_sparsity}" \
    --initial_value "${initial_value}" \
    --hln_sigma "${hln_sigma}" \
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
