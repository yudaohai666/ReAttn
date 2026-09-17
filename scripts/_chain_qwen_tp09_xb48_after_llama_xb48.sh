#!/bin/bash
# Wait for the Llama-3.1 tp0.95/xb48/mb8/full run to FULLY finish (its 128k summary
# csv exists = ground truth) AND no infer workers AND GPUs free, then launch the
# Qwen3 tp0.9/xb48/mb8/full 5-length run on 8 GPUs.
# Non-destructive: only polls, never kills anything.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
PREREQ_128K="eval/benchmarks/results/ruler/exps_llama_31_8b/reuse_v1_tp09label_evtp0.95_nohead_minblk8_xb48_lastfull/trainTP0.9_evalTP0.95_nohead_minblk8_lastfull_mb8_trainXB64_evalXB48_128k"

gpus_free () {
  local maxmem
  maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${maxmem:-999999}" -lt 20000 ]
}

echo "[chain] waiting for llama xb48 run + free GPUs ..."
while true; do
  if ls ${PREREQ_128K}/*/summary/summary_*.csv >/dev/null 2>&1 \
     && ! pgrep -f "openicl_infer.py" >/dev/null 2>&1 && gpus_free; then
    break
  fi
  sleep 60
done

echo "[chain] llama xb48 done + GPUs free at $(date -Is); starting qwen3 tp0.9/xb48"
ts=$(date +%Y%m%d_%H%M%S)
bash scripts/eval_ruler_qwen3_tp09label_all_evtp09_nohead_minblk8_xb48_lastfull.sh \
  > "logs/eval_evtp09_nohead_minblk8_xb48_lastfull_${ts}.log" 2>&1
echo "[chain] qwen3 tp0.9/xb48 finished at $(date -Is); ALL DONE"
