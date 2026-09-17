#!/bin/bash
# RULER 128k for the Qwen3 tp=0.9 mb8/xb64 trained label, 4 eval configs:
#   1) evtp0.75 per_head=1 last_q_full=1
#   2) evtp0.75 per_head=0 last_q_full=1
#   3) evtp0.8  per_head=0 last_q_full=0
#   4) evtp0.7  per_head=1 last_q_full=1
# All: MIN_BLOCKS=16, MAX_BLOCKS(eval xb)=1024. Mirrors the existing
# 8k/16k/32k/64k launchers; only the 128k point is run here.
# 8-GPU parallel; the 4 configs run sequentially.
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
export QWEN3_REUSE_V1_MAX_BLOCKS=1024
export QWEN3_REUSE_V1_MIN_BLOCKS=16
export REUSE_V1_LOG_BLOCKS=1
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"
export RULER_MAX_SEQ_LEN_K=128

RESULTS="${REPO_ROOT}/eval/benchmarks/results/ruler/exps_qwen3_8b"

# args: top_p per_head last_q_full base_dir sub_prefix tag
run_cfg () {
  export QWEN3_REUSE_V1_TOP_P="$1"
  export QWEN3_REUSE_V1_PER_HEAD_TOPP="$2"
  export QWEN3_REUSE_V1_LAST_Q_FULL="$3"
  local wd="${RESULTS}/$4/$5_128k"
  mkdir -p "${wd}"
  echo "=== [qwen3 RULER 128k $6] START $(date -Is) ==="
  echo "  top_p=$1 per_head_topp=$2 last_q_full=$3 min_blocks=16 max_blocks=1024"
  echo "  work_dir=${wd}"
  "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers ${NWORKERS} -w "${wd}"
  echo "=== [qwen3 RULER 128k $6] EXIT $? $(date -Is) ==="
}

run_cfg 0.75 1 1 \
  "reuse_v1_tp09label_evtp0.75_perhead_minblk16_xb1024_lastfull" \
  "trainTP0.9_evalTP0.75_perhead_minblk16_lastfull_mb8_trainXB64_evalXB1024" \
  "tp0.75 perhead lastfull"

run_cfg 0.75 0 1 \
  "reuse_v1_tp09label_evtp0.75_nohead_minblk16_xb1024_lastfull" \
  "trainTP0.9_evalTP0.75_nohead_minblk16_lastfull_mb8_trainXB64_evalXB1024" \
  "tp0.75 nohead lastfull"

run_cfg 0.8 0 0 \
  "reuse_v1_tp09label_evtp0.8_nohead_minblk16_xb1024_nolastfull" \
  "trainTP0.9_evalTP0.8_nohead_minblk16_nolastfull_mb8_trainXB64_evalXB1024" \
  "tp0.8 nohead nolastfull"

run_cfg 0.7 1 1 \
  "reuse_v1_tp09label_evtp0.7_perhead_minblk16_xb1024_lastfull" \
  "trainTP0.9_evalTP0.7_perhead_minblk16_lastfull_mb8_trainXB64_evalXB1024" \
  "tp0.7 perhead lastfull"

echo "=== ALL Qwen3 tp0.9-label RULER 128k 4-config DONE $(date -Is) ==="
