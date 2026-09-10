#!/bin/bash
# Llama-3.1-8B-Instruct RULER flashattn (dense baseline) @ 128k
# Dense upper bound for comparing against reuse_v1 sparse runs.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export RULER_MAX_SEQ_LEN_K=128
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
# GPUs are shared with another user's job; reduce allocator fragmentation so a
# tight-memory run has a better chance of fitting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
WORK_DIR="./results/ruler/exps_llama_31_8b/flashattn_128k"

cd "${REPO_ROOT}/eval/benchmarks"

echo "========================================="
echo "flashattn (dense) @ 128k"
echo "========================================="
"${OC}" ruler/exps_llama_31_8b/flashattn.py --max-num-workers 8 -w "${WORK_DIR}"
echo "--- 128k exit=$? ---"

echo "=== done ==="
