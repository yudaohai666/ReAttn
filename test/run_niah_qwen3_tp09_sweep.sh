#!/bin/bash
# Qwen3-8B reuse_v1 NIAH: rw=0.002/0.0025/0.003 with tp=0.9 labels (8-GPU parallel)
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
PY="${REPO_ROOT}/.venv/bin/python"
BASE="${REPO_ROOT}/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
TOP_P="${TOP_P:-0.7}"
LAST_Q_FULL="${LAST_Q_FULL:-0}"

SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

run_one_rw() {
  local rw="$1"
  # These labels were trained with tp=0.9 (not tp=0.7)
  local LBL="${BASE}/hc-orig-rw=${rw}-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt"
  [ -f "${LBL}" ] || { echo "label not found: ${LBL}" >&2; return 1; }

  local SUFFIX=""
  local EXTRA=()
  if [ "${LAST_Q_FULL}" = "1" ]; then
    SUFFIX="_lqf"
    EXTRA=(--last-q-full)
  fi

  local SUBDIR="${REPO_ROOT}/test/results/niah/Qwen3-8B_reuse_v1_rw${rw}_tp0.9trainlabel_inftp${TOP_P}${SUFFIX}"
  local LOGDIR="${SUBDIR}/logs"
  mkdir -p "${LOGDIR}"

  echo "############ rw=${rw} label_tp=0.9 inf_top_p=${TOP_P} last_q_full=${LAST_Q_FULL} -> ${SUBDIR} ############"
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
  echo "=== all shards done for rw=${rw} ==="

  "${PY}" "${REPO_ROOT}/test/show_image.py" \
    --model "${MODEL}" --eval_path "${SUBDIR}" \
    --save_dir "${REPO_ROOT}/test/results/niah/vis" \
    --expected_answer "eat a sandwich and sit in Dolores Park on a sunny day."
}

for rw in 0.002 0.0025 0.003; do
  run_one_rw "${rw}"
done

echo "=== sweep done ==="
