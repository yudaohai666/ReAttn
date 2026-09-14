#!/bin/bash
# Qwen3-8B RULER reuse_v1 @ 32k, tp=0.9 label (matched infer top_p=0.9),
# per_head_topp=False (topp coverage is true-mass / total, now the only mode).
# Runs BOTH last_q_full=0 (no last-full) and last_q_full=1 (last-full),
# into separate work_dirs so results don't clobber.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=100
export RULER_MAX_SEQ_LEN_K=32
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export QWEN3_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export QWEN3_REUSE_V1_TOP_P=0.9

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
CFG="ruler/exps_qwen3_8b/reuse_v1.py"

cd "${REPO_ROOT}/eval/benchmarks"

if [ ! -f "${QWEN3_REUSE_V1_LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${QWEN3_REUSE_V1_LABEL_PATH}" >&2
  exit 1
fi

echo "========================================="
echo "Qwen3-8B reuse_v1 tp=0.9 total @ 32k  --  last_q_full=0 (no last-full)"
echo "========================================="
QWEN3_REUSE_V1_LAST_Q_FULL=0 \
  "${OC}" "${CFG}" --max-num-workers 8 \
    -w "./results/ruler/exps_qwen3_8b/reuse_v1_total_32k"
echo "--- no-lastfull exit=$? ---"

echo "========================================="
echo "Qwen3-8B reuse_v1 tp=0.9 total @ 32k  --  last_q_full=1 (last-full)"
echo "========================================="
QWEN3_REUSE_V1_LAST_Q_FULL=1 \
  "${OC}" "${CFG}" --max-num-workers 8 \
    -w "./results/ruler/exps_qwen3_8b/reuse_v1_lastfull_total_32k"
echo "--- lastfull exit=$? ---"

echo "=== done ==="
