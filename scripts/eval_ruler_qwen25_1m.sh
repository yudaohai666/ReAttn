#!/bin/bash
# Qwen2.5-7B-Instruct-1M RULER: reuse_v1 (no lqf) -> reuse_v1 (lqf) -> flashattn
# Usage: bash eval_ruler_qwen25_1m.sh
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export QWEN25_1M_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/Qwen2.5-7B-Instruct-1M/hc-orig-rw=0.008-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_131072-multi_passkey10-sp4/label.pt

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
cd "${REPO_ROOT}/eval/benchmarks"

run_lengths() {
  local cfg="$1"
  shift
  for K in 4 8 16 32 64 128; do
    echo "  -> ${cfg} @ ${K}k"
    RULER_MAX_SEQ_LEN_K=${K} "${OC}" "$cfg" --max-num-workers 8 "$@"
  done
}

echo "========================================="
echo "[1/3] reuse_v1 (no last_q_full)"
echo "========================================="
QWEN25_1M_REUSE_V1_LAST_Q_FULL=0 run_lengths ruler/exps_qwen25_7b_1m/reuse_v1.py

echo "========================================="
echo "[2/3] reuse_v1 (last_q_full=1)"
echo "========================================="
QWEN25_1M_REUSE_V1_LAST_Q_FULL=1 run_lengths ruler/exps_qwen25_7b_1m/reuse_v1.py

echo "========================================="
echo "[3/3] flashattn baseline"
echo "========================================="
run_lengths ruler/exps_qwen25_7b_1m/flashattn.py

echo "=== all done ==="
