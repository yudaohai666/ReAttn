#!/bin/bash
# Parallel NIAH sweep for reuse_v1 + topp(0.9), one shard per GPU (8 GPUs).
# Each shard writes results_gpu<N>.json into the same result dir; show_image.py
# globs *.json in that dir, so the final heatmap merges all shards.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
PY="${REPO_ROOT}/.venv/bin/python"
LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct/hc-orig-rw=0.0013-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt"

METHOD="${METHOD:-reuse_v1}"
RUN_TAG="${RUN_TAG:-topp0.9_rw0.0013_sp0.8}"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
# last_q_full is an inference-only recall booster (no training counterpart):
# the last query block of sparse kv-heads attends densely to the full KV.
# Default OFF so the sweep matches the trained configuration; set LAST_Q_FULL=1
# to measure the boosted variant.
LAST_Q_FULL="${LAST_Q_FULL:-0}"

SUBDIR="${REPO_ROOT}/test/results/niah/$(basename ${MODEL})_${METHOD}_${RUN_TAG}"
LOGDIR="${SUBDIR}/logs"
mkdir -p "${LOGDIR}"

# Balanced shards: each GPU gets one short + one long context.
SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

COMMON=(--model "${MODEL}" --method "${METHOD}" --attn-impl sdpa
        --depths "${DEPTHS}" --max-new-tokens 50)
if [ "${METHOD}" = "reuse_v1" ]; then
  COMMON+=(--label-path "${LABEL}" --budget 32 --block-size 128 --segment-size 2048
           --sink-blocks 1 --local-blocks 2
           --select-mode topp --top-p 0.9 --min-blocks 8 --max-blocks 64)
  [ "${LAST_Q_FULL}" = "1" ] && COMMON+=(--last-q-full)
fi

echo "=== NIAH parallel: method=${METHOD} tag=${RUN_TAG} -> ${SUBDIR}"
for i in "${!SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${i}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${REPO_ROOT}/test/run_needle.py" "${COMMON[@]}" \
    --lengths "${SHARDS[$i]}" \
    --output "${SUBDIR}/results_gpu${i}.json" \
    > "${LOGDIR}/gpu${i}.log" 2>&1 &
  echo "  gpu${i}: lengths=${SHARDS[$i]} -> ${LOGDIR}/gpu${i}.log"
done
wait
echo "=== all shards done ==="

"${PY}" "${REPO_ROOT}/test/show_image.py" \
  --model "${MODEL}" --eval_path "${SUBDIR}" \
  --save_dir "${REPO_ROOT}/test/results/niah/vis" \
  --expected_answer "eat a sandwich and sit in Dolores Park on a sunny day."
echo "=== heatmap under test/results/niah/vis/$(basename ${MODEL})/ ==="
