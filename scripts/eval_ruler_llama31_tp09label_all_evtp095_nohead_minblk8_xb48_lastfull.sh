#!/bin/bash
# RULER 8k+16k+32k+64k+128k for Llama-3.1-8B-Instruct, reuse_v1 tp=0.9 mb8/xb64 label.
# Setting: TOP_P=0.95, eval max_blocks(xb)=48, MIN_BLOCKS=8, per_head_topp=0
# (nohead / group-shared nucleus), last_q_full=1.
# REUSE_V1_LOG_BLOCKS=1 prints per-layer selected-block stats to worker logs.
# Each context point gets its own opencompass --work-dir. 8 parallel GPU workers.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
cd "${REPO_ROOT}/eval/benchmarks"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NWORKERS="${NWORKERS:-8}"

export TIKTOKEN_CACHE_DIR="${REPO_ROOT}/.cache/tiktoken"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export LLAMA_REUSE_V1_LABEL_PATH="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
export LLAMA_REUSE_V1_TOP_P=0.95
export LLAMA_REUSE_V1_MAX_BLOCKS=48
export LLAMA_REUSE_V1_MIN_BLOCKS=8
export LLAMA_REUSE_V1_PER_HEAD_TOPP=0
export LLAMA_REUSE_V1_LAST_Q_FULL=1

export REUSE_V1_LOG_BLOCKS=1
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"

BASE_WD="${REPO_ROOT}/eval/benchmarks/results/ruler/exps_llama_31_8b/reuse_v1_tp09label_evtp0.95_nohead_minblk8_xb48_lastfull"

run () {
  local k="$1"
  local wd="${BASE_WD}/trainTP0.9_evalTP0.95_nohead_minblk8_lastfull_mb8_trainXB64_evalXB48_${k}k"
  export RULER_MAX_SEQ_LEN_K="${k}"
  mkdir -p "${wd}"
  echo "=== [llama3.1 RULER ${k}k tp0.95 nohead minblk8 lastfull evalXB48] START $(date -Is) ==="
  echo "  top_p=0.95 per_head_topp=0 min_blocks=8 max_blocks=48 last_q_full=1 log_blocks=1"
  echo "  work_dir=${wd}"
  "${OC}" ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers ${NWORKERS} -w "${wd}"
  echo "=== [llama3.1 RULER ${k}k tp0.95 nohead minblk8 lastfull evalXB48] EXIT $? $(date -Is) ==="
}

run 8
run 16
run 32
run 64
run 128
echo "=== ALL Llama3.1 reuse_v1 RULER 8k+16k+32k+64k+128k tp0.95 nohead minblk8 xb48 lastfull DONE $(date -Is) ==="
