#!/bin/bash
# Qwen2.5-7B-Instruct-1M reuse_v1 NIAH: rw=0.008, top_p=0.7 (8-GPU parallel)
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen2.5-7B-Instruct-1M"
PY="${REPO_ROOT}/.venv/bin/python"
LBL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen2.5-7B-Instruct-1M/hc-orig-rw=0.008-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_131072-multi_passkey10-sp4/label.pt"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
TOP_P="${TOP_P:-0.7}"
LAST_Q_FULL="${LAST_Q_FULL:-0}"

[ -f "${LBL}" ] || { echo "label not found: ${LBL}" >&2; exit 1; }

SUFFIX=""
EXTRA=()
if [ "${LAST_Q_FULL}" = "1" ]; then
  SUFFIX="_lqf"
  EXTRA=(--last-q-full)
fi

SUBDIR="${REPO_ROOT}/test/results/niah/Qwen2.5-7B-1M_reuse_v1_rw0.008_tp${TOP_P}${SUFFIX}"
LOGDIR="${SUBDIR}/logs"
mkdir -p "${LOGDIR}"

SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

echo "=== Qwen2.5-7B-1M reuse_v1 rw=0.008 top_p=${TOP_P} last_q_full=${LAST_Q_FULL} -> ${SUBDIR} ==="
for i in "${!SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${i}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${REPO_ROOT}/test/run_needle.py" \
      --model "${MODEL}" --method reuse_v1 --attn-impl sdpa \
      --depths "${DEPTHS}" --max-new-tokens 50 \
      --label-path "${LBL}" --budget 32 --block-size 128 --segment-size 2048 \
      --sink-blocks 1 --local-blocks 2 \
      --select-mode topp --top-p "${TOP_P}" --min-blocks 8 --max-blocks 64 \
      "${EXTRA[@]+"${EXTRA[@]}"}" \
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
echo "=== heatmap under test/results/niah/vis/Qwen2.5-7B-Instruct-1M/ ==="
