#!/bin/bash
# Llama-3.1-8B-Instruct RULER reuse_v1: rw=0.003 tp=0.9 label (2026-09-10 retrain,
# short-path attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/...), inference topp=0.9.
# MATCHED config: label top_p == inference top_p == 0.9. last_q_full=1, 128k only.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export RULER_MAX_SEQ_LEN_K=128
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LLAMA_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export LLAMA_REUSE_V1_TOP_P=0.9
export LLAMA_REUSE_V1_LAST_Q_FULL=1

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
# Tagged work_dir: tp0.9 label + tp0.9 inference (matched). Distinct from the
# tp0.95, tp0.7_v2, and label-tp0.7v2_infer-tp0.9 runs so predictions never collide.
WORK_DIR="./results/ruler/exps_llama_31_8b/reuse_v1_lastfull_tp0.9"

cd "${REPO_ROOT}/eval/benchmarks"

if [ ! -f "${LLAMA_REUSE_V1_LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${LLAMA_REUSE_V1_LABEL_PATH}" >&2
  exit 1
fi

echo "========================================="
echo "reuse_v1 tp=0.9 (matched label+infer) last_q_full=1 @ 128k"
echo "========================================="
"${OC}" ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8 -w "${WORK_DIR}"
echo "--- 128k exit=$? ---"

echo "=== done ==="
