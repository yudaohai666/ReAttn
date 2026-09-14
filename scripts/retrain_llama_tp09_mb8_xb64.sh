#!/bin/bash
# Train the tp=0.9 counterpart of the Llama-3.1-8B mb8/xb64 label.
# Identical to the tp=0.8 run (rw=0.003, init=0.0, sp=0.8, mb8, xb64,
# EXP_TAG=mb8-xb64) except top_p 0.8 -> 0.9. Uses all 8 GPUs (sp_size=8),
# so it gates on no in-flight train_reuse.py and >=90GiB free per GPU first.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/logs/retrain_topp09_mb8_xb64"
mkdir -p "${LOGDIR}"

LLAMA="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
NEED_FREE_MIB=92160

echo "=== [wait] any in-flight train_reuse.py to finish ($(date -Is)) ==="
while pgrep -f "reuse_v1/train_reuse.py" >/dev/null 2>&1; do sleep 30; done
echo "  none running $(date -Is)"

echo "=== [gate] waiting for >=${NEED_FREE_MIB} MiB free on all 8 ($(date -Is)) ==="
stable=0
while true; do
  min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if [ "${min_free}" -ge "${NEED_FREE_MIB}" ]; then
    stable=$((stable+1)); echo "  min_free=${min_free} MiB OK (${stable}/2) $(date -Is)"
    [ "${stable}" -ge 2 ] && break
  else
    stable=0; echo "  min_free=${min_free} MiB busy... $(date -Is)"
  fi
  sleep 30
done

echo "=== [train] START llama-tp0.9-mb8-xb64 $(date -Is) ==="
EXP_TAG=mb8-xb64 bash scripts/train_reuse_hc.sh \
  "${LLAMA}" 8000 128000 0.01 10 8 0.003 0.0 0.8 0.9 8 64 \
  > "${LOGDIR}/llama_tp09_mb8_xb64.log" 2>&1
echo "=== [train] EXIT $? llama-tp0.9-mb8-xb64 $(date -Is) ==="
