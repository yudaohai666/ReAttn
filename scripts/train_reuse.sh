#!/usr/bin/env bash
# Train the reuse_v1 DuoAttention-style per-(layer, kv-head) anchor/sparse gate.
#
# Mirrors duo-attention/scripts/train.sh but drives reuse_v1/train_reuse.py:
#   - Ulysses sequence parallelism (--sp_size) + FSDP2, both on the full world
#     mesh: one sequence is split across sp_size ranks (each holds
#     seq_len/sp_size tokens), attention all-to-alls internally so every rank
#     still sees the WHOLE sequence for its head subset -> global top-k block
#     selection stays EXACT. dp_size = 8/sp_size groups run distinct samples.
#   - two-pass forward (--two_pass): no_grad [teacher/student] "fill" passes +
#     grad "use" pass; activation checkpointing + SP let 128k fit on 8xH800 at
#     ~12.8GB/rank with no CPU offload.
#
# Llama-3.1-8B has 32 q-heads / 8 kv-heads, both divisible by 8 -> sp_size=8
# (dp_size=1) on a single 8-GPU node. sp_size=1 falls back to pure FSDP2 DP
# (each rank runs the full sequence; OOMs at 128k -- use sp_size=8 there).
#
# The reuse_v1 hyperparameters (budget=32, block_size=128, segment_size=2048,
# sink_blocks=1, local_blocks=2, select_mode=topk) are LOCKED inside
# reuse_v1/train_reuse.py and MUST match inference, so they are NOT script args.
#
# Usage:
#   bash scripts/train_reuse.sh <model_path> <ctx_len_min> <ctx_len_max> <reg_weight> <lr> <num_passkey> [sp_size] [reg_mode] [desired_density]
#
#   reg_mode        "l1" (default): deterministic gate + L1 sparsity (<reg_weight>
#                   applies). "kuma": HardKuma stochastic gate + adaptive-lambda
#                   Lagrangian density constraint (<reg_weight> is IGNORED; target
#                   sparsity is driven by <desired_density> instead).
#   desired_density kuma only: target full-model anchor (dense-head) fraction
#                   (layer 0 included, frozen all-anchor); lower -> sparser. The
#                   lagrange_lr (0.01) / lamda_init (0.5) stay at the reference
#                   defaults. Ignored in l1 mode.
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
reg_weight=${4}
lr=${5}
num_passkey=${6}
sp_size=${7:-8}
reg_mode=${8:-l1}
# kuma default matches the reference train_kuma_multi_passkey.sh (very aggressive
# target; layer 0 is frozen all-anchor and still counts in this full-model mean).
desired_density=${9:-0.08333}

if [ "${reg_mode}" = "kuma" ]; then
    # kuma: reg_weight is ignored; identify the run by its density target.
    setting="kuma-dens=${desired_density}-lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
    reg_args=(--reg_mode kuma --desired_density "${desired_density}")
else
    setting="lr=${lr}-reg=${reg_weight}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
    reg_args=(--reg_mode l1)
fi
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
    --reg_weight "${reg_weight}" \
    "${reg_args[@]}" \
    --initial_value 1.0 \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min "${ctx_len_min}" \
    --context_length_max "${ctx_len_max}" \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkey "${num_passkey}" \
    --save_steps 50 \
    --two_pass \
    --sp_size "${sp_size}" \
    --output_dir "attn_patterns/${exp_name}"


