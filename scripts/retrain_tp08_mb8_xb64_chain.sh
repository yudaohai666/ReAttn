#!/bin/bash
# Chained launcher: train mb8/xb64 top_p=0.8 for Qwen3 then Llama, sequentially.
# Each run uses all 8 GPUs (sp_size=8), so they cannot overlap. Waits for any
# in-flight train_reuse.py (the current tp=0.9 Qwen3 run) to finish, then gates
# on GPU free memory before each launch. EXP_TAG=mb8-xb64 forces the explicit
# tag even though 8/64 is the script default (otherwise no suffix is added).
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/logs/retrain_topp08_mb8_xb64"
mkdir -p "${LOGDIR}"

QWEN3="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
LLAMA="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
NEED_FREE_MIB=92160

wait_no_training () {
  echo "=== [wait] for any in-flight train_reuse.py to finish ($(date -Is)) ==="
  while pgrep -f "reuse_v1/train_reuse.py" >/dev/null 2>&1; do
    sleep 30
  done
  echo "  no training running $(date -Is)"
}

gate_gpu () {
  local stable=0
  echo "=== [gate] waiting for >=${NEED_FREE_MIB} MiB free on all 8 ($(date -Is)) ==="
  while true; do
    local min_free
    min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "${min_free}" -ge "${NEED_FREE_MIB}" ]; then
      stable=$((stable+1))
      echo "  min_free=${min_free} MiB OK (${stable}/2) $(date -Is)"
      [ "${stable}" -ge 2 ] && break
    else
      stable=0
      echo "  min_free=${min_free} MiB busy, waiting... $(date -Is)"
    fi
    sleep 30
  done
}

run_one () {
  local model="$1" tag="$2" log="$3"
  wait_no_training
  gate_gpu
  echo "=== [train] START ${tag} $(date -Is) ==="
  EXP_TAG=mb8-xb64 bash scripts/train_reuse_hc.sh \
    "${model}" 8000 128000 0.01 10 8 0.003 0.0 0.8 0.8 8 64 \
    > "${log}" 2>&1
  echo "=== [train] EXIT $? ${tag} $(date -Is) ==="
}

run_one "${QWEN3}" "qwen3-tp0.8-mb8-xb64" "${LOGDIR}/qwen3_tp08_mb8_xb64.log"
run_one "${LLAMA}" "llama-tp0.8-mb8-xb64" "${LOGDIR}/llama_tp08_mb8_xb64.log"
echo "=== ALL tp=0.8 mb8/xb64 TRAINING DONE $(date -Is) ==="
