#!/bin/bash
# Qwen3-8B RULER reuse_v1: rw=0.003 tp=0.9 label
# (attn_patterns/reuse_v1/Qwen3-8B/...tp=0.9...), inference topp=0.9.
# MATCHED config: label top_p == inference top_p == 0.9. last_q_full=1, 128k only.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export RULER_MAX_SEQ_LEN_K=128
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export QWEN3_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export QWEN3_REUSE_V1_TOP_P=0.9
export QWEN3_REUSE_V1_LAST_Q_FULL=1

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
# Tagged work_dir: tp0.9 label + tp0.9 inference (matched), Qwen3.
WORK_DIR="./results/ruler/exps_qwen3_8b/reuse_v1_lastfull_tp0.9"

cd "${REPO_ROOT}/eval/benchmarks"

if [ ! -f "${QWEN3_REUSE_V1_LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${QWEN3_REUSE_V1_LABEL_PATH}" >&2
  exit 1
fi

echo "========================================="
echo "Qwen3-8B reuse_v1 tp=0.9 (matched label+infer) last_q_full=1 @ 128k"
echo "========================================="
"${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8 -w "${WORK_DIR}"
echo "--- 128k exit=$? ---"

echo "=== done ==="
