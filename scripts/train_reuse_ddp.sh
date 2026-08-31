#!/usr/bin/env bash
# Train the reuse_v1 per-(layer, kv-head) anchor/sparse gate with Deterministic
# Differentiable Pruning (--reg_mode ddp).
#
# This is the ddp-only entrypoint: the command line carries ONLY ddp-relevant
# args. The L1 path lives in scripts/train_reuse.sh, the HardKuma path in
# scripts/train_reuse_kuma.sh -- use THIS script when you always want ddp.
#
# DDP mechanism (aligned with yellowtree123/Deterministic-Differentiable-Pruning):
#   anchor_heads IS the learnable logit z_loga (init ~N(1.0, z_loga_init_std) so
#   the gate starts dense-side, sigmoid(1.0)~0.73, pruning toward target sparsity);
#   the blend gate = straight-through clamp(z_loga, 0, 1). A
#   3-term LEARNABLE-lambda Lagrangian on the annealed soft-saturation score drives
#   the expected SPARSE fraction (1 - score.mean()) toward --target_sparsity via
#   dual ascent (lambda_1 linear 1e-2, lambda_2 quadratic 1e-2, lambda_3
#   binary-entropy 1e-2; all maximize=True). soft-saturation mean holds
#   at 0.5 for the first 10% of steps (anneal_warmup_ratio=0.1) then anneals
#   0.5 -> 0.1 (sqrt), giving the primal time to move before the sigmoid sharpens.
#   Export zeroes the exact round(target_sparsity * L * H) lowest-scoring heads
#   (global top-k). Layer 0 stays all-anchor (frozen, excluded).
#   --reg_weight and --desired_density are IGNORED here (l1 / kuma only).
#
# The reuse_v1 inference hyperparameters (budget=32, block_size=128,
# segment_size=2048, sink_blocks=1, local_blocks=2) are LOCKED inside
# reuse_v1/train_reuse.py and MUST match inference, so they are NOT args.
# Block selection here is TOPP (nucleus: top_p=0.9, min_blocks=8, max_blocks=64);
# budget=32 is only the cache-fallback width. INFERENCE MUST use the SAME topp
# params or the exported head label will not transfer.
#
# Usage:
#   bash scripts/train_reuse_ddp.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [target_sparsity]
#
#   sp_size         Ulysses SP group size. Default 8 (Llama-3.1-8B 32q/8kv, 128k
#                   fits ~12.8GB/rank). sp_size=1 = pure FSDP2 DP (OOMs at 128k).
#   target_sparsity target fraction of (layer>=1) kv-heads that end up SPARSE;
#                   higher -> sparser label. Default 0.7695 (DDP reference).
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
lambda_1_lr=${8:-0.005}
lambda_2_lr=${9:-0.05}
lambda_3_lr=${10:-0.005}
anneal_warmup_ratio=${11:-0.35}
initial_value=${12:-0.5}
lambda_init_value=${13:-1.0}
no_ac=${14:-1}   # 1 = 关闭 AC（默认关闭，仅 sp_size>1 时安全）
setting="ddp-sp=${target_sparsity}-lr=${lr}-l1lr=${lambda_1_lr}-l2lr=${lambda_2_lr}-wu=${anneal_warmup_ratio}-init=${initial_value}-linit=${lambda_init_value}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
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
    --reg_mode ddp \
    --target_sparsity "${target_sparsity}" \
    --lambda_1_lr "${lambda_1_lr}" \
    --lambda_2_lr "${lambda_2_lr}" \
    --lambda_3_lr "${lambda_3_lr}" \
    --select_mode topp \
    --top_p 0.9 \
    --min_blocks 8 \
    --max_blocks 64 \
    --initial_value "${initial_value}" \
    --z_loga_init_std 1e-2 \
    --lambda_init_value "${lambda_init_value}" \
    --anneal_schedule sqrt \
    --anneal_mean_min 0.1 \
    --anneal_warmup_ratio "${anneal_warmup_ratio}" \
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
