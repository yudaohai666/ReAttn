#!/bin/bash
# Wait for the tp0.95/xb64 nolastfull run to FULLY finish (its 128k summary csv
# exists = ground truth, survives log cleanup) AND no infer workers running AND
# GPUs free, then launch the tp0.95/xb128 nolastfull 5-length run.
# Non-destructive: only polls, never kills anything.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
XB64_128K="eval/benchmarks/results/ruler/exps_qwen3_8b/reuse_v1_tp09label_evtp0.95_nohead_minblk16_xb64_nolastfull/trainTP0.9_evalTP0.95_nohead_minblk16_nolastfull_mb8_trainXB64_evalXB64_128k"

gpus_free () {
  # returns 0 if every GPU uses < 20000 MiB
  local maxmem
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${maxmem:-999999}" -lt 20000 ]
}

while true; do
  done_xb64=0
  ls ${XB64_128K}/*/summary/summary_*.csv >/dev/null 2>&1 && done_xb64=1
  if [ "${done_xb64}" = "1" ] && ! pgrep -f "openicl_infer.py" >/dev/null 2>&1 && gpus_free; then
    break
  fi
  sleep 60
done

echo "[chain] tp0.95/xb64 done + GPUs free at $(date -Is); starting tp0.95/xb128"
ts=$(date +%Y%m%d_%H%M%S)
setsid bash scripts/eval_ruler_qwen3_tp09label_all_evtp095_nohead_minblk16_xb128_nolastfull.sh \
  > "logs/eval_evtp095_nohead_minblk16_xb128_nolastfull_${ts}.log" 2>&1
echo "[chain] tp0.95/xb128 launched (log ts=${ts}) at $(date -Is)"
