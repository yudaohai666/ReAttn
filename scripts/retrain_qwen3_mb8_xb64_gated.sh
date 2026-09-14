#!/bin/bash
# Gated launcher: reproduce the original -912 Qwen3 config (min_blocks=8,
# max_blocks=64 == the script default, so exp_name gets NO suffix) but with the
# fixed topp code. Waits until all 8 shared GPUs free up (~89GB/card needed),
# then launches on all 8. Output dir: attn_patterns/reuse_v1/Qwen3-8B/hc-orig-...-sp8
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/logs/retrain_topp_mb8_xb64"
mkdir -p "${LOGDIR}"

QWEN3="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
# Need ~89GB/card; require >=92GB free on every GPU, twice in a row, before launch.
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

echo "=== [train] START mb8-xb64 (original config + topp fix) $(date -Is) ==="
bash scripts/train_reuse_hc.sh \
  "${QWEN3}" 8000 128000 0.01 10 8 0.003 0.0 0.8 0.9 8 64 \
  > "${LOGDIR}/qwen3_mb8_xb64.log" 2>&1
echo "=== [train] EXIT $? $(date -Is) ==="
