#!/usr/bin/env bash
# Train the reuse_v1 per-(layer, kv-head) anchor/sparse gate with the HardKuma
# stochastic gate + adaptive-Lagrangian density constraint (--reg_mode kuma).
#
# This is the kuma-only entrypoint: the command line carries ONLY kuma-relevant
# args. The L1 path lives in scripts/train_reuse.sh (which also has a positional
# reg_weight that kuma ignores) -- use THIS script when you always want kuma.
#
# Fixed to the reference train_kuma_multi_passkey.sh values (NOT exposed here):
#   lamda_init=2.0, lagrange_lr=0.001, lagrange_alpha=0.9, lambda clamp [1e-12,20]
#   (all come from reuse_v1/utils.py argparse defaults / train_reuse.py constants).
# The reuse_v1 inference hyperparameters (budget=32, block_size=128,
# segment_size=2048, sink_blocks=1, local_blocks=2) are LOCKED inside
# reuse_v1/train_reuse.py and MUST match inference, so they are NOT args.
# Block selection here is TOPP (nucleus: top_p=0.9, min_blocks=8, max_blocks=64);
# budget=32 is only the cache-fallback width. INFERENCE MUST use the SAME topp
# params or the exported head label will not transfer.
#
# density counts ALL layers (layer 0 frozen all-anchor, contributes ~0.9167/L to
# the mean), matching the reference. At an aggressive desired_density the frozen
# layer-0 mass eats a large slice of the target, so the trainable-layer budget is
# tighter than the reference -- raise desired_density if the gates collapse.
#
# Usage:
#   bash scripts/train_reuse_kuma.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [desired_density] [run_tag] [top_p]
#
#   sp_size         Ulysses SP group size. Default 8 (Llama-3.1-8B 32q/8kv, 128k
#                   fits ~12.8GB/rank). sp_size=1 = pure FSDP2 DP (OOMs at 128k).
#   desired_density target full-model anchor (dense-head) fraction; lower ->
#                   sparser label. Default 0.08333 (reference; very aggressive).
#   top_p           Nucleus coverage for topp block selection. Default 0.9.
#                   Lower (e.g. 0.7) -> larger sparse/dense gap -> stronger gate
#                   gradient -> faster polarization. MUST match inference top_p.
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
desired_density=${7:-0.08333}
run_tag=${8:-}
top_p=${9:-0.9}

setting="kuma-dens=${desired_density}-lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}${run_tag:+-${run_tag}}"
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
    --reg_mode kuma \
    --desired_density "${desired_density}" \
    --select_mode topp \
    --top_p "${top_p}" \
    --min_blocks 8 \
    --max_blocks 64 \
    --initial_value 1.0 \
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
    --sp_size "${sp_size}" \
    --output_dir "attn_patterns/${exp_name}"
