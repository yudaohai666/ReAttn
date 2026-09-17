#!/bin/bash
# Wait for the running tp0.9 minblk16 xb64 eval (bash pid arg) to finish,
# then launch the tp0.75 variant. Non-destructive: only polls, never kills.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
WAIT_PID="$1"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[chain] tp0.9 run (pid ${WAIT_PID}) finished at $(date -Is); starting tp0.75"
cd "${REPO_ROOT}"
ts=$(date +%Y%m%d_%H%M%S)
bash scripts/eval_ruler_qwen3_tp09label_all_evtp075_nohead_minblk16_xb64_lastfull.sh \
  > "logs/eval_evtp075_nohead_minblk16_xb64_${ts}.log" 2>&1
echo "[chain] tp0.75 run finished at $(date -Is)"
