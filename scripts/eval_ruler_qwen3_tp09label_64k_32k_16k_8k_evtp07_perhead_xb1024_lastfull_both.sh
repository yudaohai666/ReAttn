#!/bin/bash
# RULER 64k+32k+16k+8k for the Qwen3 tp=0.9 mb8/xb64 trained label.
# Inference TOP_P=0.7, eval xb (max_blocks)=1024, PER-HEAD topp ON, run BOTH
# last_q_full ON and OFF.
# per_head_topp: each q-head in a GQA group picks its own nucleus (no amax over
# the group); topp-only, costs G x IndexCache memory. Enabled via
# QWEN3_REUSE_V1_PER_HEAD_TOPP=1.
# REUSE_V1_LOG_BLOCKS=1 prints per-layer selected-block stats to worker logs.
# 8 sequential runs = {lastfull,nolastfull} x {64k,32k,16k,8k}, each its own
# opencompass --work-dir so results are tagged and never clobber each other.
# 8 parallel GPU workers.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
cd "${REPO_ROOT}/eval/benchmarks"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# RULER cwe/fwe/vt build contexts with tiktoken (cl100k_base). Offline node:
# point tiktoken at the pre-downloaded blob cache.
export TIKTOKEN_CACHE_DIR="${REPO_ROOT}/.cache/tiktoken"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export QWEN3_REUSE_V1_LABEL_PATH="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
export QWEN3_REUSE_V1_TOP_P=0.7
export QWEN3_REUSE_V1_MAX_BLOCKS=1024
export QWEN3_REUSE_V1_MIN_BLOCKS=8
export QWEN3_REUSE_V1_PER_HEAD_TOPP=1

# Per-layer block-count logging (evaluation aid).
export REUSE_V1_LOG_BLOCKS=1

export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"

BASE_WD="${REPO_ROOT}/eval/benchmarks/results/ruler/exps_qwen3_8b/reuse_v1_tp09label_evtp0.7_perhead_xb1024"

run () {
  local lastfull="$1"
  local k="$2"
  local lf_tag; [ "${lastfull}" = "1" ] && lf_tag="lastfull" || lf_tag="nolastfull"
  local wd="${BASE_WD}/trainTP0.9_evalTP0.7_perhead_${lf_tag}_mb8_trainXB64_evalXB1024_${k}k"
  export QWEN3_REUSE_V1_LAST_Q_FULL="${lastfull}"
  export RULER_MAX_SEQ_LEN_K="${k}"
  mkdir -p "${wd}"
  echo "=== [qwen3 RULER ${k}k tp0.7 perhead ${lf_tag} evalXB1024] START $(date -Is) ==="
  echo "  top_p=0.7 per_head_topp=1 max_blocks=1024 min_blocks=8 last_q_full=${lastfull} log_blocks=1"
  echo "  work_dir=${wd}"
  "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8 -w "${wd}"
  echo "=== [qwen3 RULER ${k}k tp0.7 perhead ${lf_tag} evalXB1024] EXIT $? $(date -Is) ==="
}

for lf in 1 0; do
  for k in 64 32 16 8; do
    run "${lf}" "${k}"
  done
done
echo "=== ALL Qwen3 tp0.9-label RULER 64k+32k+16k+8k tp0.7 perhead xb1024 lastfull+nolastfull DONE $(date -Is) ==="
