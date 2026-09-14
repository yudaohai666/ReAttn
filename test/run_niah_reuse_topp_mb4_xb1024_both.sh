#!/bin/bash
# Sequential NIAH sweeps for the corrected-topp reuse_v1 labels (min_blocks=4,
# max_blocks=1024 == unlimited). Llama first (all 8 GPUs), then Qwen3 (all 8 GPUs).
# Uses the parallel sweep harness (2 lengths/GPU). Detached via setsid by the caller.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/test/results—910/niah_mb4_xb1024_logs"
mkdir -p "${LOGDIR}"

export TOP_P=0.9 MIN_BLOCKS=4 MAX_BLOCKS=1024 SELECT_MODE=topp METHOD=reuse_v1

LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb4-xb1024/label.pt"
QW_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb4-xb1024/label.pt"

echo "=== [llama] START $(date -Is) ==="
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct" \
LABEL="${LL_LABEL}" \
RUN_TAG="topp0.9_rw0.003_sp0.8_mb4_xb1024" \
  bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/llama.log" 2>&1
echo "=== [llama] EXIT $? $(date -Is) ==="

echo "=== [qwen3] START $(date -Is) ==="
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B" \
LABEL="${QW_LABEL}" \
RUN_TAG="topp0.9_rw0.003_sp0.8_mb4_xb1024" \
  bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/qwen3.log" 2>&1
echo "=== [qwen3] EXIT $? $(date -Is) ==="
echo "=== ALL NIAH DONE $(date -Is) ==="
