#!/usr/bin/env bash
# Train reuse_v1 with Hard Concrete + topk_ratio block selection.
#
# Block budget per q-block = ceil(causal_valid_k * topk_ratio),
# clamped to [min_blocks, nkb].  Adaptive sparsity: short prefixes get
# fewer blocks, long prefixes scale proportionally.
#
# Usage:
#   bash scripts/train_reuse_hc_topkratio.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [reg_weight] [initial_value] [target_sparsity] [topk_ratio] [min_blocks]
#
#   sp_size          Ulysses SP group size. Default 8.
#   reg_weight       L0 penalty coefficient. Default 0.003.
#   initial_value    Initial log_alpha. Default 0.0.
#   target_sparsity  Fraction of heads to export as sparse (top-k cutoff). Default 0.8.
#   topk_ratio       Dynamic block budget ratio. Default 0.05.
#   min_blocks       Minimum blocks per q-block (hard floor). Default 8.
set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

model_name=${1}
ctx_len_min=${2}
ctx_len_max=${3}
lr=${4}
num_passkey=${5}
sp_size=${6:-8}
reg_weight=${7:-0.003}
initial_value=${8:-0.0}
target_sparsity=${9:-0.8}
topk_ratio=${10:-0.05}
min_blocks=${11:-8}

setting="hc-topkratio-rw=${reg_weight}-init=${initial_value}-sp=${target_sparsity}-tkr=${topk_ratio}-mb=${min_blocks}-lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
exp_name="reuse_v1/$(basename ${model_name})/${setting}"

echo "=== reuse_v1 HC + topk_ratio training ==="
echo "  model:           ${model_name}"
echo "  ctx:             ${ctx_len_min} ~ ${ctx_len_max}"
echo "  sp_size:         ${sp_size}"
echo "  reg_weight:      ${reg_weight}"
echo "  target_sparsity: ${target_sparsity}"
echo "  topk_ratio:      ${topk_ratio}"
echo "  min_blocks:      ${min_blocks}"
echo "  output:          attn_patterns/${exp_name}"
echo ""

VENV_TORCHRUN="${SCRIPT_DIR}/../.venv/bin/torchrun"
TORCHRUN="${VENV_TORCHRUN}"

"${TORCHRUN}" --nnodes 1 --nproc_per_node 8 \
    reuse_v1/train_reuse.py \
    --model_name "${model_name}" \
    --batch_size 1 \
    --max_length "${ctx_len_max}" \
    --dataset_name "datasets/PaulGrahamEssays.jsonl" \
    --dataset_format "multiple_passkey" \
    --num_steps 2000 \
    --lr "${lr}" \
    --reg_mode hc \
    --reg_weight "${reg_weight}" \
    --initial_value "${initial_value}" \
    --target_sparsity "${target_sparsity}" \
    --select_mode topk_ratio \
    --topk_ratio "${topk_ratio}" \
    --min_blocks "${min_blocks}" \
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
    --disable_wandb \
    --sp_size "${sp_size}" \
    --output_dir "attn_patterns/${exp_name}"
