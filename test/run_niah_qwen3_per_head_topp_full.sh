#!/bin/bash
# Full NIAH sweep for Qwen3-8B reuse_v1 (tp=0.9 label, top_p=0.9), comparing
# per_head_topp=False (shared amax) vs per_head_topp=True (per q-head nucleus).
# All lengths (8k..128k, 16 shards-values) x all depths (10). 8 GPUs per mode;
# the two modes run sequentially. Heatmaps rendered per mode at the end.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
PY="${REPO_ROOT}/.venv/bin/python"
LBL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt"
[ -f "${LBL}" ] || { echo "label not found: ${LBL}" >&2; exit 1; }

DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
TOP_P="${TOP_P:-0.9}"
LAST_Q_FULL="${LAST_Q_FULL:-0}"   # set 1 to enable last-q-full recall booster (both modes)

# Balanced shards: each GPU gets one short + one long context.
SHARDS=(
  "8192,131072" "16384,122880" "24576,114688" "32768,106496"
  "40960,98304" "49152,90112"  "57344,81920"  "65536,73728"
)

COMMON=(--model "${MODEL}" --method reuse_v1 --attn-impl sdpa
        --depths "${DEPTHS}" --max-new-tokens 50
        --label-path "${LBL}" --budget 32 --block-size 128 --segment-size 2048
        --sink-blocks 1 --local-blocks 2
        --select-mode topp --top-p "${TOP_P}" --min-blocks 8 --max-blocks 64)
[ "${LAST_Q_FULL}" = "1" ] && COMMON+=(--last-q-full)

run_mode() {
  local tag="$1"; shift          # extra args (e.g. --per-head-topp)
  local SUBDIR="${REPO_ROOT}/test/results/niah/Qwen3-8B_reuse_v1_tp0.9_${tag}"
  local LOGDIR="${SUBDIR}/logs"
  mkdir -p "${LOGDIR}"
  echo "=== mode=${tag} -> ${SUBDIR}  (extra: $*)"
  for i in "${!SHARDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${i}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${REPO_ROOT}/test/run_needle.py" "${COMMON[@]}" "$@" \
        --lengths "${SHARDS[$i]}" \
        --output "${SUBDIR}/results_gpu${i}.json" \
        > "${LOGDIR}/gpu${i}.log" 2>&1 &
    echo "  gpu${i}: lengths=${SHARDS[$i]}"
  done
  wait
  echo "=== mode=${tag} shards done ==="
  "${PY}" "${REPO_ROOT}/test/show_image.py" \
    --model "${MODEL}" --eval_path "${SUBDIR}" \
    --save_dir "${REPO_ROOT}/test/results/niah/vis" \
    --expected_answer "eat a sandwich and sit in Dolores Park on a sunny day."
}

run_mode "shared"                                   # per_head_topp=False (default)
run_mode "perhead" --per-head-topp                  # per_head_topp=True

echo "=== DONE. heatmaps under test/results/niah/vis/Qwen3-8B/ ==="
