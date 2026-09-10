#!/usr/bin/env bash
# Train reuse_v1 anchor/sparse labels for Qwen2.5-7B-Instruct-1M (Hard Concrete)
# using topk_ratio selection mode (dynamic per-q-block budget).
#
# Block budget per q-block = ceil(causal_valid_k * TOPK_RATIO), clamped to
# [MIN_BLOCKS, nkb].  This produces adaptive sparsity: short prefixes get a
# tighter budget while long prefixes approach TOPK_RATIO * context blocks.
#
# Key differences vs Llama / Qwen3:
#   - num_attention_heads=28, num_key_value_heads=4  -> sp_size MUST be 4
#     (valid divisors of both 28 and 4: 1, 2, 4)
#   - dp_size = 8 / sp_size = 2  (each GPU pair shares one sequence)
#   - dual_chunk_attention_config is neutralized in train_reuse.py
#   - rope_theta=1e7 is already in model config; no rope override needed
#
# Usage:
#   bash scripts/run_train_reuse_qwen25_1m_topkratio.sh
#   Override any variable via env:
#     TOPK_RATIO=0.1 MIN_BLOCKS=16 bash scripts/run_train_reuse_qwen25_1m_topkratio.sh
set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen2.5-7B-Instruct-1M}"
CTX_MIN="${CTX_MIN:-8000}"
CTX_MAX="${CTX_MAX:-131072}"
LR="${LR:-0.01}"
NUM_PASSKEY="${NUM_PASSKEY:-10}"
SP_SIZE="${SP_SIZE:-4}"          # MUST divide both num_attention_heads=28 and num_kv_heads=4
REG_WEIGHT="${REG_WEIGHT:-0.002}"
INITIAL_VALUE="${INITIAL_VALUE:-0.0}"
TARGET_SPARSITY="${TARGET_SPARSITY:-0.8}"
TOPK_RATIO="${TOPK_RATIO:-0.05}"
MIN_BLOCKS="${MIN_BLOCKS:-8}"

setting="hc-topkratio-rw=${REG_WEIGHT}-init=${INITIAL_VALUE}-sp=${TARGET_SPARSITY}-tkr=${TOPK_RATIO}-mb=${MIN_BLOCKS}-lr=${LR}-ctx=${CTX_MIN}_${CTX_MAX}-multi_passkey${NUM_PASSKEY}-sp${SP_SIZE}"
exp_name="reuse_v1/$(basename ${MODEL})/${setting}"

echo "=== Qwen2.5-7B-Instruct-1M reuse_v1 training (topk_ratio) ==="
echo "  model:           ${MODEL}"
echo "  ctx:             ${CTX_MIN} ~ ${CTX_MAX}"
echo "  sp_size:         ${SP_SIZE}  (dp_size = $((8 / SP_SIZE)))"
echo "  reg_weight:      ${REG_WEIGHT}"
echo "  target_sparsity: ${TARGET_SPARSITY}"
echo "  topk_ratio:      ${TOPK_RATIO}"
echo "  min_blocks:      ${MIN_BLOCKS}"
echo "  output:          attn_patterns/${exp_name}"
echo ""

VENV_TORCHRUN="${SCRIPT_DIR}/../.venv/bin/torchrun"
TORCHRUN="${VENV_TORCHRUN}"

MASTER_PORT="${MASTER_PORT:-29511}"

"${TORCHRUN}" --nnodes 1 --nproc_per_node 8 --master_port "${MASTER_PORT}" \
    reuse_v1/train_reuse.py \
    --model_name "${MODEL}" \
    --batch_size 1 \
    --max_length "${CTX_MAX}" \
    --dataset_name "datasets/PaulGrahamEssays.jsonl" \
    --dataset_format "multiple_passkey" \
    --num_steps 2000 \
    --lr "${LR}" \
    --reg_mode hc \
    --reg_weight "${REG_WEIGHT}" \
    --initial_value "${INITIAL_VALUE}" \
    --target_sparsity "${TARGET_SPARSITY}" \
    --select_mode topk_ratio \
    --topk_ratio "${TOPK_RATIO}" \
    --min_blocks "${MIN_BLOCKS}" \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min "${CTX_MIN}" \
    --context_length_max "${CTX_MAX}" \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkeys "${NUM_PASSKEY}" \
    --save_steps 50 \
    --two_pass \
    --no_ac \
    --sp_size "${SP_SIZE}" \
    --output_dir "attn_patterns/${exp_name}"
