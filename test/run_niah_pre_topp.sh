#!/bin/bash
# NIAH validation for a PREVIOUSLY trained label under attn_patterns_pre/
# (label file is named after its directory, not label.pt), with reuse_v1 + topp(0.9).
#
# Usage:
#   bash test/run_niah_pre_topp.sh            # full 8-GPU sharded sweep
#   SMOKE=1 bash test/run_niah_pre_topp.sh    # single case, 1 GPU, quick sanity
#
# SMOKE=1 exists to confirm the label passes reuse_prefill.load_label's
# layer-0-all-anchor assertion before spending a full sweep on it.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}"
PY="${REPO_ROOT}/.venv/bin/python"

RUN_NAME="hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8"
LABEL="${LABEL:-${REPO_ROOT}/attn_patterns_pre/Llama-3.1-8B-Instruct/${RUN_NAME}/${RUN_NAME}.pt}"

RUN_TAG="${RUN_TAG:-pre_topp0.9_rw0.0013_sp0.8}"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
# Match the top_p the label was TRAINED with (see the run's config.json).
TOP_P="${TOP_P:-0.9}"
# Inference-only recall booster; OFF so the sweep matches the trained config.
LAST_Q_FULL="${LAST_Q_FULL:-0}"

[ -f "${LABEL}" ] || { echo "label not found: ${LABEL}" >&2; exit 1; }

SUBDIR="${REPO_ROOT}/test/results/niah/$(basename ${MODEL})_reuse_v1_${RUN_TAG}"
LOGDIR="${SUBDIR}/logs"
mkdir -p "${LOGDIR}"

COMMON=(--model "${MODEL}" --method reuse_v1 --attn-impl sdpa
        --depths "${DEPTHS}" --max-new-tokens 50
        --label-path "${LABEL}" --budget 32 --block-size 128 --segment-size 2048
        --sink-blocks 1 --local-blocks 2
        --select-mode topp --top-p "${TOP_P}" --min-blocks 8 --max-blocks 64)
[ "${LAST_Q_FULL}" = "1" ] && COMMON+=(--last-q-full)

echo "=== label: ${LABEL}"
echo "=== top_p: ${TOP_P}"
echo "=== out:   ${SUBDIR}"

if [ "${SMOKE:-0}" = "1" ]; then
  SMOKE_LEN="${SMOKE_LEN:-16384}"
  SMOKE_DEPTH="${SMOKE_DEPTH:-50}"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${REPO_ROOT}/test/run_needle.py" \
      --model "${MODEL}" --method reuse_v1 --attn-impl sdpa \
      --lengths "${SMOKE_LEN}" --depths "${SMOKE_DEPTH}" --max-new-tokens 50 \
      --label-path "${LABEL}" --budget 32 --block-size 128 --segment-size 2048 \
      --sink-blocks 1 --local-blocks 2 \
      --select-mode topp --top-p "${TOP_P}" --min-blocks 8 --max-blocks 64 \
      --output "${SUBDIR}/smoke_len${SMOKE_LEN}_d${SMOKE_DEPTH}.json"
  echo "=== smoke done -> ${SUBDIR}/smoke_len${SMOKE_LEN}_d${SMOKE_DEPTH}.json"
  exit 0
fi

# Balanced shards: each GPU gets one short + one long context.
SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

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
