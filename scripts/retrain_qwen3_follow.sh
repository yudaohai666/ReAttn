#!/usr/bin/env bash
# Follower: wait for the in-flight Llama retrain (torchrun) to finish, then run
# the Qwen3-8B retrain on the freed 8 GPUs with identical corrected-topp params
# (min_blocks=4, max_blocks=1024, tp=0.9, sp8). Sequential by design.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/logs/retrain_topp_mb4_xb1024"
mkdir -p "${LOGDIR}"

LLAMA_TORCHRUN_PID="${1:-58735}"
QWEN3="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"

echo "=== [follower] waiting for llama torchrun pid=${LLAMA_TORCHRUN_PID} $(date -Is) ==="
while kill -0 "${LLAMA_TORCHRUN_PID}" 2>/dev/null; do
  sleep 60
done
echo "=== [follower] llama torchrun gone; settling GPUs $(date -Is) ==="
sleep 30

echo "=== [qwen3] START $(date -Is) ==="
bash scripts/train_reuse_hc.sh \
  "${QWEN3}" 8000 128000 0.01 10 8 0.003 0.0 0.8 0.9 4 1024 \
  > "${LOGDIR}/qwen3.log" 2>&1
echo "=== [qwen3] EXIT $? $(date -Is) ==="
echo "=== ALL RETRAIN DONE $(date -Is) ==="
