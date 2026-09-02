#!/bin/bash
# Dense FlashAttention-2 baseline on NIAH, same 8-GPU shard layout as the
# reuse_v1 sweeps so the numbers are directly comparable.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}"
PY="${REPO_ROOT}/.venv/bin/python"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"

SUBDIR="${REPO_ROOT}/test/results/niah/$(basename ${MODEL})_flashattn"
LOGDIR="${SUBDIR}/logs"
mkdir -p "${LOGDIR}"

SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

echo "=== dense FlashAttention-2 baseline -> ${SUBDIR}"
for i in "${!SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${i}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${REPO_ROOT}/test/run_needle.py" \
      --model "${MODEL}" --method flashattn --attn-impl flash_attention_2 \
      --depths "${DEPTHS}" --max-new-tokens 50 \
      --lengths "${SHARDS[$i]}" \
      --output "${SUBDIR}/results_gpu${i}.json" \
      > "${LOGDIR}/gpu${i}.log" 2>&1 &
  echo "  gpu${i}: lengths=${SHARDS[$i]}"
done
wait
echo "=== all shards done ==="

"${PY}" "${REPO_ROOT}/test/show_image.py" \
  --model "${MODEL}" --eval_path "${SUBDIR}" \
  --save_dir "${REPO_ROOT}/test/results/niah/vis" \
  --expected_answer "eat a sandwich and sit in Dolores Park on a sunny day."
