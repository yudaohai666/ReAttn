#!/bin/bash
# Llama-3.1-8B-Instruct RULER: flashattn (dense) 8/16/32/64k  +  xattention 8/16/32/64/128k
# flashattn 128k already done separately -> results/ruler/exps_llama_31_8b/flashattn_128k
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
# GPUs are shared with another user's job; reduce allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
cd "${REPO_ROOT}/eval/benchmarks"

for K in 8 16 32 64; do
  echo "========================================="
  echo "flashattn (dense) @ ${K}k"
  echo "========================================="
  RULER_MAX_SEQ_LEN_K=${K} "${OC}" ruler/exps_llama_31_8b/flashattn.py \
    --max-num-workers 8 -w "./results/ruler/exps_llama_31_8b/flashattn_${K}k"
  echo "--- flashattn ${K}k exit=$? ---"
done

for K in 8 16 32 64 128; do
  echo "========================================="
  echo "xattention @ ${K}k"
  echo "========================================="
  RULER_MAX_SEQ_LEN_K=${K} "${OC}" ruler/exps_llama_31_8b/xattn.py \
    --max-num-workers 8 -w "./results/ruler/exps_llama_31_8b/xattn_${K}k"
  echo "--- xattn ${K}k exit=$? ---"
done

echo "=== done ==="
