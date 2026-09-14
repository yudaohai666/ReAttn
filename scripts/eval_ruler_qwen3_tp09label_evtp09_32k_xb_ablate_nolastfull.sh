#!/bin/bash
# RULER 32k xb (max_blocks) ablation for the Qwen3 tp=0.9 mb8/xb64 trained label.
# Inference TOP_P=0.9, NO last_q_full (LAST_Q_FULL=0), REUSE_V1_LOG_BLOCKS=1
# prints per-layer selected-block stats to the worker logs.
# Sweeps eval xb in {64, 96, 128, 1024}; each xb gets its own opencompass
# --work-dir so results are tagged and never clobber each other. All 5 RULER
# task groups at a single 32k context point, 8 parallel GPU workers, sequential.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
cd "${REPO_ROOT}/eval/benchmarks"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# RULER cwe/fwe/vt build contexts with tiktoken (cl100k_base). This node has no
# internet egress, so point tiktoken at the pre-downloaded blob cache.
export TIKTOKEN_CACHE_DIR="${REPO_ROOT}/.cache/tiktoken"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export QWEN3_REUSE_V1_LABEL_PATH="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
export QWEN3_REUSE_V1_TOP_P=0.9
export QWEN3_REUSE_V1_MIN_BLOCKS=8
export QWEN3_REUSE_V1_LAST_Q_FULL=0

# Per-layer block-count logging (evaluation aid).
export REUSE_V1_LOG_BLOCKS=1

export RULER_MAX_SEQ_LEN_K="${RULER_MAX_SEQ_LEN_K:-32}"
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"

# Base dir for this ablation; each xb run gets a distinct tagged sub-work_dir.
BASE_WD="${REPO_ROOT}/eval/benchmarks/results/ruler/exps_qwen3_8b/reuse_v1_tp09label_evtp0.9_nolastfull_${RULER_MAX_SEQ_LEN_K}k"

run () {
  local xb="$1"
  local wd="${BASE_WD}/trainTP0.9_evalTP0.9_mb8_trainXB64_evalXB${xb}"
  export QWEN3_REUSE_V1_MAX_BLOCKS="${xb}"
  mkdir -p "${wd}"
  echo "=== [qwen3 RULER ${RULER_MAX_SEQ_LEN_K}k evalXB${xb}] START $(date -Is) ==="
  echo "  label       = ${QWEN3_REUSE_V1_LABEL_PATH}"
  echo "  top_p       = ${QWEN3_REUSE_V1_TOP_P}  max_blocks = ${QWEN3_REUSE_V1_MAX_BLOCKS}  min_blocks = ${QWEN3_REUSE_V1_MIN_BLOCKS}"
  echo "  last_q_full = ${QWEN3_REUSE_V1_LAST_Q_FULL}  log_blocks = ${REUSE_V1_LOG_BLOCKS}"
  echo "  work_dir    = ${wd}"
  "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8 -w "${wd}"
  echo "=== [qwen3 RULER ${RULER_MAX_SEQ_LEN_K}k evalXB${xb}] EXIT $? $(date -Is) ==="
}

for xb in 64 96 128 1024; do
  run "${xb}"
done
echo "=== ALL Qwen3 tp0.9-label RULER ${RULER_MAX_SEQ_LEN_K}k xb ablation DONE $(date -Is) ==="
