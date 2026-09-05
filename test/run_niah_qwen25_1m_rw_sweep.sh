#!/bin/bash
# Qwen2.5-7B-Instruct-1M reuse_v1 NIAH sweep (8-GPU parallel)
# Label path pattern: attn_patterns/reuse_v1/Qwen2.5-7B-Instruct-1M/hc-orig-rw=<rw>-...-sp4/label.pt
#
# Key differences vs Qwen3/Llama:
#   - num_attention_heads=28, num_key_value_heads=4 (sp4 labels)
#   - label dirname ends with -sp4 (trained with sp_size=4)
#   - dual_chunk_attention_config is auto-neutralized in run_needle.py
#
# Usage:
#   bash run_niah_qwen25_1m_rw_sweep.sh            # all rw values sequentially
#   RW=0.003 bash run_niah_qwen25_1m_rw_sweep.sh   # single rw value
#   LAST_Q_FULL=1 bash run_niah_qwen25_1m_rw_sweep.sh  # enable last-q-full boost
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen2.5-7B-Instruct-1M"
PY="${REPO_ROOT}/.venv/bin/python"
BASE="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen2.5-7B-Instruct-1M"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
TOP_P="${TOP_P:-0.7}"
LAST_Q_FULL="${LAST_Q_FULL:-0}"

# 8 shards covering 8k~131k, each GPU takes a disjoint length range
SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

run_one_rw() {
  local rw="$1"
  # Label trained with sp_size=4 -> dirname ends with -sp4
  local LBL="${BASE}/hc-orig-rw=${rw}-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_131072-multi_passkey10-sp4/label.pt"
  [ -f "${LBL}" ] || { echo "label not found: ${LBL}" >&2; return 1; }

  local SUFFIX=""
  local EXTRA=()
  if [ "${LAST_Q_FULL}" = "1" ]; then
    SUFFIX="_lqf"
    EXTRA=(--last-q-full)
  fi

  local SUBDIR="${REPO_ROOT}/test/results/niah/Qwen2.5-7B-1M_reuse_v1_rw${rw}_tp${TOP_P}${SUFFIX}"
  local LOGDIR="${SUBDIR}/logs"
  mkdir -p "${LOGDIR}"

  echo "############ rw=${rw} top_p=${TOP_P} last_q_full=${LAST_Q_FULL} -> ${SUBDIR} ############"
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

if [ -n "${RW:-}" ]; then
  run_one_rw "${RW}"
else
  for rw in 0.002 0.003 0.004 0.005; do
    run_one_rw "${rw}"
  done
fi

echo "=== sweep done ==="
