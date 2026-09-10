#!/bin/bash
# Qwen3-8B RULER reuse_v1: rw=0.003 tp=0.9 label, inference topp=0.9 (matched).
# Sweep 8/16/32/64k. Chains AFTER the in-flight 128k run so they don't contend
# for the same 8 GPUs. last_q_full=1.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export QWEN3_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export QWEN3_REUSE_V1_TOP_P=0.9
export QWEN3_REUSE_V1_LAST_Q_FULL=1

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
WAIT_LOG="${REPO_ROOT}/logs/ruler_qwen3_tp09_128k_20260910_160657.log"

# --- wait for the in-flight 128k run to finish ---
if [ -f "${WAIT_LOG}" ]; then
  echo "waiting for qwen3 tp0.9 128k run to finish..."
  while ! tail -c 200 "${WAIT_LOG}" | grep -q "=== done ==="; do
    sleep 60
  done
  echo "128k finished, starting 8/16/32/64k sweep at $(date)"
fi

cd "${REPO_ROOT}/eval/benchmarks"

if [ ! -f "${QWEN3_REUSE_V1_LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${QWEN3_REUSE_V1_LABEL_PATH}" >&2
  exit 1
fi

for K in 8 16 32 64; do
  echo "========================================="
  echo "Qwen3-8B reuse_v1 tp=0.9 (matched) last_q_full=1 @ ${K}k"
  echo "========================================="
  RULER_MAX_SEQ_LEN_K=${K} "${OC}" ruler/exps_qwen3_8b/reuse_v1.py \
    --max-num-workers 8 -w "./results/ruler/exps_qwen3_8b/reuse_v1_lastfull_tp0.9_${K}k"
  echo "--- ${K}k exit=$? ---"
done

echo "=== done ==="
