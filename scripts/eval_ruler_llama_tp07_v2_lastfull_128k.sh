#!/bin/bash
# Llama-3.1-8B-Instruct RULER reuse_v1: rw=0.003 tp=0.7 label (2026-09-09 retrain,
# short-path attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/...), inference topp=0.7
# last_q_full=1 (same setting as eval_ruler_lastfull.sh), 128k only
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export RULER_MAX_SEQ_LEN_K=128
# Shared tiktoken cache: avoids 8 workers each re-downloading cl100k_base
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken

export LLAMA_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export LLAMA_REUSE_V1_TOP_P=0.7
export LLAMA_REUSE_V1_LAST_Q_FULL=1

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
# Separate work_dir: this label differs from the older nested-path tp=0.7 label
# (same hyper-params, 4 anchor heads different), so predictions must not be shared.
WORK_DIR="./results/ruler/exps_llama_31_8b/reuse_v1_lastfull_tp0.7_v2"

cd "${REPO_ROOT}/eval/benchmarks"

if [ ! -f "${LLAMA_REUSE_V1_LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${LLAMA_REUSE_V1_LABEL_PATH}" >&2
  exit 1
fi

echo "========================================="
echo "reuse_v1 tp=0.7 (v2 label) last_q_full=1 @ 128k"
echo "========================================="
"${OC}" ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8 -w "${WORK_DIR}"
echo "--- 128k exit=$? ---"

echo "=== done ==="
