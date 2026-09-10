#!/bin/bash
# Llama-3.1-8B-Instruct RULER PBS-Attn baseline @ 8/16/32/64/128k
# Waits for the in-flight flashattn+xattention sweep to finish first, so the two
# runs don't contend for the same 8 GPUs.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
SWEEP_LOG="${REPO_ROOT}/logs/ruler_llama_flash_xattn_sweep_20260909_223906.log"

# --- wait for the flashattn+xattn sweep to print its terminating marker ---
if [ -f "${SWEEP_LOG}" ]; then
  echo "waiting for flashattn+xattn sweep to finish..."
  while ! tail -c 200 "${SWEEP_LOG}" | grep -q "=== done ==="; do
    sleep 60
  done
  echo "sweep finished, starting PBS at $(date)"
fi

cd "${REPO_ROOT}/eval/benchmarks"

for K in 8 16 32 64 128; do
  echo "========================================="
  echo "PBS-Attn @ ${K}k"
  echo "========================================="
  RULER_MAX_SEQ_LEN_K=${K} "${OC}" ruler/exps_llama_31_8b/pbs.py \
    --max-num-workers 8 -w "./results/ruler/exps_llama_31_8b/pbs_${K}k"
  echo "--- pbs ${K}k exit=$? ---"
done

echo "=== done ==="
