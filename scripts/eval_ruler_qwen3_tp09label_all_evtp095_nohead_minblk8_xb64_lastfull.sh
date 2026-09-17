#!/bin/bash
# RULER 8k+16k+32k+64k+128k for the Qwen3 tp=0.9 mb8/xb64 trained label.
# Variant of evtp0.9_xb64_lastfull with MIN_BLOCKS=16 (base uses 8).
# Setting: TOP_P=0.9, eval xb (max_blocks)=64, MIN_BLOCKS=16, per_head_topp=0
# (nohead / group-shared nucleus), last_q_full=1.
# REUSE_V1_LOG_BLOCKS=1 prints per-layer selected-block stats to worker logs.
# Each context point gets its own opencompass --work-dir. 8 parallel GPU workers.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
cd "${REPO_ROOT}/eval/benchmarks"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NWORKERS=8

export TIKTOKEN_CACHE_DIR="${REPO_ROOT}/.cache/tiktoken"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export QWEN3_REUSE_V1_LABEL_PATH="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
export QWEN3_REUSE_V1_TOP_P=0.95
export QWEN3_REUSE_V1_MAX_BLOCKS=64
export QWEN3_REUSE_V1_MIN_BLOCKS=8
export QWEN3_REUSE_V1_PER_HEAD_TOPP=0
export QWEN3_REUSE_V1_LAST_Q_FULL=1

export REUSE_V1_LOG_BLOCKS=1
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"

BASE_WD="${REPO_ROOT}/eval/benchmarks/results/ruler/exps_qwen3_8b/reuse_v1_tp09label_evtp0.95_nohead_minblk8_xb64_lastfull"

run () {
  local k="$1"
  local wd="${BASE_WD}/trainTP0.9_evalTP0.95_nohead_minblk8_lastfull_mb8_trainXB64_evalXB64_${k}k"
  export RULER_MAX_SEQ_LEN_K="${k}"
  mkdir -p "${wd}"
  echo "=== [qwen3 RULER ${k}k tp0.95 nohead minblk8 lastfull evalXB64] START $(date -Is) ==="
  echo "  top_p=0.95 per_head_topp=0 min_blocks=8 max_blocks=64 last_q_full=1 log_blocks=1"
  echo "  work_dir=${wd}"
  "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers ${NWORKERS} -w "${wd}"
  echo "=== [qwen3 RULER ${k}k tp0.95 nohead minblk8 lastfull evalXB64] EXIT $? $(date -Is) ==="
}

run 8
run 16
run 32
run 64
run 128
echo "=== ALL Qwen3 tp0.9-label RULER 8k+16k+32k+64k+128k tp0.95 nohead minblk8 xb64 lastfull DONE $(date -Is) ==="
