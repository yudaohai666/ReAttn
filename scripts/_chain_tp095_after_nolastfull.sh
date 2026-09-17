#!/bin/bash
# Wait for the current tp0.9 nolastfull 8/16/32k run to finish (via its DONE
# marker + no live infer workers), then launch tp0.95 nolastfull 5-length run.
# Non-destructive: only polls, never kills.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
CUR_LOG="/"
cd "${REPO_ROOT}"
while true; do
  if grep -q "nolastfull DONE" "${CUR_LOG}" 2>/dev/null && ! pgrep -f "openicl_infer.py" >/dev/null 2>&1; then
    break
  fi
  sleep 60
done
echo "[chain] tp0.9 nolastfull finished at $(date -Is); starting tp0.95"
ts=$(date +%Y%m%d_%H%M%S)
bash scripts/eval_ruler_qwen3_tp09label_all_evtp095_nohead_minblk16_xb64_nolastfull.sh \
  > "logs/eval_evtp095_nohead_minblk16_xb64_nolastfull_${ts}.log" 2>&1
echo "[chain] tp0.95 run finished at $(date -Is)"
