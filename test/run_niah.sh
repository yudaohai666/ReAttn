#!/bin/bash
# NIAH (needle-in-a-haystack) for the pbs-attn sparse-attention methods.
# Defaults to reuse_v1 (pure-Triton cross-layer block-sparse reuse, H800-capable).
# Switch method via env, e.g. METHOD=xattention bash run_niah.sh
# Output under test/results/niah/.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# PY="${PY:-${REPO_ROOT}/.venv/bin/python}"

PY="${PY:-${REPO_ROOT}/.venv/bin/python}"


GPU="${GPU:-0}"
MODEL="${MODEL:-/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct}"
MODEL_NAME=$(basename "${MODEL}")
BASE_RESULT_DIR="${SCRIPT_DIR}/results/niah"
VIS_DIR="${BASE_RESULT_DIR}/vis"

LENGTHS="${LENGTHS:-8192,16384,24576,32768,40960,49152,57344,65536,73728,81920,90112,98304,106496,114688,122880,131072}"
DEPTHS="${DEPTHS:-0,11,22,33,44,56,67,78,89,100}"
EXPECTED_ANSWER="eat a sandwich and sit in Dolores Park on a sunny day."

METHOD="${METHOD:-reuse_v1}"           # reuse_v1 | flashattn | xattention | sparse_reuse | ...
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
# reuse_v1 is a prefill-patch method: decode uses the original forward. The
# project venv has no flash-attn built, so default the decode path to sdpa.
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

# Per-method knobs (override via env).
STRIDE="${STRIDE:-8}"; THRESHOLD="${THRESHOLD:-0.9}"
GAMMA="${GAMMA:-0.95}"; TAU="${TAU:-0.1}"
LABEL_PATH="${LABEL_PATH:-${REPO_ROOT}/ckp/Llama-3.1-8B-Instruct/full_head.pt}"
SPARSE_PHASE="${SPARSE_PHASE:-prefill}"; SINK_BLOCKS="${SINK_BLOCKS:-1}"
LOCAL_BLOCKS="${LOCAL_BLOCKS:-2}"; TOPK="${TOPK:-32}"
DENSE_TAIL="${DENSE_TAIL:-0}"
BUDGET="${BUDGET:-32}"; BLOCK_SIZE="${BLOCK_SIZE:-128}"; SEGMENT_SIZE="${SEGMENT_SIZE:-2048}"
# reuse_v1 block-selection: SELECT_MODE=topk|topp; topp knobs TOP_P/MIN_BLOCKS/MAX_BLOCKS
# (MIN/MAX_BLOCKS are inclusive of sink+local; MAX_BLOCKS empty -> kernel headroom).
SELECT_MODE="${SELECT_MODE:-topk}"; TOP_P="${TOP_P:-0.9}"
MIN_BLOCKS="${MIN_BLOCKS:-8}"; MAX_BLOCKS="${MAX_BLOCKS:-64}"
# LAST_Q_FULL=1: sparse kv-heads' last query block attends densely to full KV
# (matches the RULER / LongBench eval default).
LAST_Q_FULL="${LAST_Q_FULL:-0}"
# duo (DuoAttention): retrieval heads dense flash_attn + streaming heads sink+local.
DUO_LABEL_DIR="${DUO_LABEL_DIR:-${REPO_ROOT}/ckp/duo/Llama-3.1-8B-Instruct}"
SINK_SIZE="${SINK_SIZE:-128}"; RECENT_SIZE="${RECENT_SIZE:-256}"; SPARSITY="${SPARSITY:-0.5}"

case "${METHOD}" in
  xattention)  ARGS=(--stride "${STRIDE}" --threshold "${THRESHOLD}") ;;
  flexprefill) ARGS=(--gamma "${GAMMA}" --tau "${TAU}") ;;
  duo)
    ARGS=(--attn-load-dir "${DUO_LABEL_DIR}" --sink-size "${SINK_SIZE}" \
          --recent-size "${RECENT_SIZE}" --sparsity "${SPARSITY}" \
          --block-size "${BLOCK_SIZE}") ;;
  flashattn)   ARGS=() ;;
  sparse_reuse)
    ARGS=(--label-path "${LABEL_PATH}" --sparse-phase "${SPARSE_PHASE}" \
          --sink-blocks "${SINK_BLOCKS}" --local-blocks "${LOCAL_BLOCKS}" --topk "${TOPK}" \
          --dense-tail "${DENSE_TAIL}") ;;
  reuse_v1)
    ARGS=(--label-path "${LABEL_PATH}" --budget "${BUDGET}" \
          --block-size "${BLOCK_SIZE}" --segment-size "${SEGMENT_SIZE}" \
          --sink-blocks "${SINK_BLOCKS}" --local-blocks "${LOCAL_BLOCKS}" \
          --select-mode "${SELECT_MODE}" --top-p "${TOP_P}" --min-blocks "${MIN_BLOCKS}")
    [ -n "${MAX_BLOCKS}" ] && ARGS+=(--max-blocks "${MAX_BLOCKS}")
    [ "${LAST_Q_FULL}" = "1" ] && ARGS+=(--last-q-full) ;;
  *) ARGS=() ;;
esac

SUBDIR="${BASE_RESULT_DIR}/${MODEL_NAME}_${METHOD}"
# DuoAttention runs vary by sparsity -> keep each sparsity's results separate.
[ "${METHOD}" = "duo" ] && SUBDIR="${SUBDIR}_sparsity${SPARSITY}"
# Optional per-run tag to keep parallel runs (e.g. different labels) separate.
[ -n "${RUN_TAG}" ] && SUBDIR="${SUBDIR}_${RUN_TAG}"
mkdir -p "${SUBDIR}"
echo "=== NIAH: method=${METHOD} -> ${SUBDIR}/results.json ==="

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${SCRIPT_DIR}/run_needle.py" \
  --model "${MODEL}" --method "${METHOD}" --attn-impl "${ATTN_IMPL}" \
  --lengths "${LENGTHS}" --depths "${DEPTHS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${SUBDIR}/results.json" "${ARGS[@]}"

"${PY}" "${SCRIPT_DIR}/show_image.py" \
  --model "${MODEL}" --eval_path "${SUBDIR}" \
  --save_dir "${VIS_DIR}" --expected_answer "${EXPECTED_ANSWER}"

echo "All done. Image under: ${VIS_DIR}/${MODEL_NAME}/"
