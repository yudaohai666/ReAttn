#!/bin/bash
# Gated launcher: wait until the shared GPUs free up (another job currently holds
# ~157GB/card), then start the Qwen3 mb8-xb1024 retrain on all 8 GPUs.
# min_blocks=8, max_blocks=1024 (unlimited). Tag auto -> ...-sp8-mb8-xb1024.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/logs/retrain_topp_mb8_xb1024"
mkdir -p "${LOGDIR}"

QWEN3="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
# Need ~80GB/card; require >=92GB free on every GPU, twice in a row, before launch.
NEED_FREE_MIB=92160
STABLE=0

echo "=== [gate] waiting for GPU memory to free ($(date -Is)); need >=${NEED_FREE_MIB} MiB free on all 8 ==="
while true; do
  min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if [ "${min_free}" -ge "${NEED_FREE_MIB}" ]; then
    STABLE=$((STABLE+1))
    echo "  min_free=${min_free} MiB OK (${STABLE}/2) $(date -Is)"
    [ "${STABLE}" -ge 2 ] && break
  else
    STABLE=0
    echo "  min_free=${min_free} MiB busy, waiting... $(date -Is)"
  fi
  sleep 30
done

echo "=== [train] START mb8-xb1024 $(date -Is) ==="
bash scripts/train_reuse_hc.sh \
  "${QWEN3}" 8000 128000 0.01 10 8 0.003 0.0 0.8 0.9 8 1024 \
  > "${LOGDIR}/qwen3_mb8_xb1024.log" 2>&1
echo "=== [train] EXIT $? $(date -Is) ==="
