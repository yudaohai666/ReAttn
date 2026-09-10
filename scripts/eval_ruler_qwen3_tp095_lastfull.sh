#!/bin/bash
# Qwen3-8B RULER reuse_v1: rw=0.003 tp=0.95 label, inference topp=0.95
# last_q_full=1 (same setting as eval_ruler_lastfull.sh), 8k/16k/32k/64k/128k
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
# Shared tiktoken cache: avoids 8 workers each re-downloading cl100k_base
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken

export QWEN3_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.95-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export QWEN3_REUSE_V1_TOP_P=0.95
export QWEN3_REUSE_V1_LAST_Q_FULL=1

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
# Isolated work_dir so tp=0.95 predictions never mix with tp=0.7/0.9 runs
WORK_DIR="./results/ruler/exps_qwen3_8b/reuse_v1_lastfull_tp0.95"

cd "${REPO_ROOT}/eval/benchmarks"

if [ ! -f "${QWEN3_REUSE_V1_LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${QWEN3_REUSE_V1_LABEL_PATH}" >&2
  exit 1
fi

for K in 8 16 32 64 128; do
  echo "========================================="
  echo "reuse_v1 tp=0.95 last_q_full=1 @ ${K}k"
  echo "========================================="
  RULER_MAX_SEQ_LEN_K=${K} "${OC}" ruler/exps_qwen3_8b/reuse_v1.py \
    --max-num-workers 8 -w "${WORK_DIR}"
  echo "--- ${K}k exit=$? ---"
done

echo "=== done ==="
