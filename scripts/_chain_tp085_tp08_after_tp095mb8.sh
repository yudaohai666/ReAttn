#!/bin/bash
# Wait for the tp0.95/mb8/xb64/full run to FULLY finish (its 128k summary csv
# exists = ground truth) AND no infer workers AND GPUs free, then run
# tp0.85 then tp0.8 (both mb16/xb64/nohead/full, 5 lengths) sequentially.
# Non-destructive: only polls, never kills anything.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
PREREQ_128K="eval/benchmarks/results/ruler/exps_qwen3_8b/reuse_v1_tp09label_evtp0.95_nohead_minblk8_xb64_lastfull/trainTP0.9_evalTP0.95_nohead_minblk8_lastfull_mb8_trainXB64_evalXB64_128k"

gpus_free () {
  local maxmem
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${maxmem:-999999}" -lt 20000 ]
}

wait_until_ready () {
  while true; do
    if ls ${PREREQ_128K}/*/summary/summary_*.csv >/dev/null 2>&1 \
       && ! pgrep -f "openicl_infer.py" >/dev/null 2>&1 && gpus_free; then
      return 0
    fi
    sleep 60
  done
}

echo "[chain] waiting for tp0.95/mb8 run + free GPUs ..."
wait_until_ready
echo "[chain] prereq done + GPUs free at $(date -Is); starting tp0.85"
ts=$(date +%Y%m%d_%H%M%S)
bash scripts/eval_ruler_qwen3_tp09label_all_evtp085_nohead_minblk16_xb64_lastfull.sh \
  > "logs/eval_evtp085_nohead_minblk16_xb64_lastfull_${ts}.log" 2>&1
echo "[chain] tp0.85 finished at $(date -Is); starting tp0.8"
ts=$(date +%Y%m%d_%H%M%S)
bash scripts/eval_ruler_qwen3_tp09label_all_evtp08_nohead_minblk16_xb64_lastfull.sh \
  > "logs/eval_evtp08_nohead_minblk16_xb64_lastfull_${ts}.log" 2>&1
echo "[chain] tp0.8 finished at $(date -Is); ALL DONE"
