#!/bin/bash
# RULER 32k ablation for the Qwen3 tp=0.9 mb8/xb64 trained label.
# REUSE_V1_LOG_BLOCKS=1 prints per-layer selected-block stats to worker logs.
# Sweeps eval xb (max_blocks) in {64, 96, 128, 1024} across three settings:
#   (A) TOP_P=0.9, last_q_full=1
#   (B) TOP_P=0.8, last_q_full=1
#   (C) TOP_P=0.8, last_q_full=0
# = 12 sequential 8-worker RULER runs. Each (topp,lastfull,xb) gets its own
# opencompass --work-dir so results are tagged and never clobber each other.
# Complements the earlier TOP_P=0.9 / last_q_full=0 xb sweep.
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
export QWEN3_REUSE_V1_MIN_BLOCKS=8

# Per-layer block-count logging (evaluation aid).
export REUSE_V1_LOG_BLOCKS=1

export RULER_MAX_SEQ_LEN_K="${RULER_MAX_SEQ_LEN_K:-32}"
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"

BASE_WD="${REPO_ROOT}/eval/benchmarks/results/ruler/exps_qwen3_8b/reuse_v1_tp09label_xb_ablate_${RULER_MAX_SEQ_LEN_K}k"

run () {
  local topp="$1" lastfull="$2" xb="$3"
  local lf_tag; [ "${lastfull}" = "1" ] && lf_tag="lastfull" || lf_tag="nolastfull"
  local wd="${BASE_WD}/trainTP0.9_evalTP${topp}_${lf_tag}_mb8_trainXB64_evalXB${xb}"
  export QWEN3_REUSE_V1_TOP_P="${topp}"
  export QWEN3_REUSE_V1_LAST_Q_FULL="${lastfull}"
  export QWEN3_REUSE_V1_MAX_BLOCKS="${xb}"
  mkdir -p "${wd}"
  echo "=== [qwen3 RULER ${RULER_MAX_SEQ_LEN_K}k tp${topp} ${lf_tag} evalXB${xb}] START $(date -Is) ==="
  echo "  top_p=${topp} max_blocks=${xb} min_blocks=8 last_q_full=${lastfull} log_blocks=1"
  echo "  work_dir=${wd}"
  "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8 -w "${wd}"
  echo "=== [qwen3 RULER ${RULER_MAX_SEQ_LEN_K}k tp${topp} ${lf_tag} evalXB${xb}] EXIT $? $(date -Is) ==="
}

# (A) TOP_P=0.9, last_q_full=1
for xb in 64 96 128 1024; do run 0.9 1 "${xb}"; done
# (B) TOP_P=0.8, last_q_full=1
for xb in 64 96 128 1024; do run 0.8 1 "${xb}"; done
# (C) TOP_P=0.8, last_q_full=0
for xb in 64 96 128 1024; do run 0.8 0 "${xb}"; done

echo "=== ALL Qwen3 tp0.9-label RULER ${RULER_MAX_SEQ_LEN_K}k xb/topp/lastfull ablation DONE $(date -Is) ==="
