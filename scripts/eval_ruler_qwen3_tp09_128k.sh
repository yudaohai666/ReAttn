#!/bin/bash
# Qwen3-8B RULER reuse_v1: rw=0.003 tp=0.9 label, inf topp=0.9
# 128k only: no last_q_full -> last_q_full=1 (sequential)
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export RULER_MAX_SEQ_LEN_K=128
export QWEN3_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt
export QWEN3_REUSE_V1_TOP_P=0.9

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
cd "${REPO_ROOT}/eval/benchmarks"

echo "========================================="
echo "[1/2] reuse_v1 no last_q_full @ 128k"
echo "========================================="
QWEN3_REUSE_V1_LAST_Q_FULL=0 "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8

echo "========================================="
echo "[2/2] reuse_v1 last_q_full=1 @ 128k"
echo "========================================="
QWEN3_REUSE_V1_LAST_Q_FULL=1 "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8

echo "=== done ==="
