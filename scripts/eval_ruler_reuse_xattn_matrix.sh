#!/bin/bash
# RULER matrix: Qwen3-8B + Llama-3.1-8B-Instruct, methods {reuse_v1, xattention},
# lengths {8k,16k,32k,64k,128k}, 200 samples.
#
# reuse_v1 config: tp=0.9 label, top_p=0.9, per_head_topp=False (topp coverage
#                  is true-mass / total, now the only mode),
#                  last_q_full=1 (last-full; the better variant established at 32k).
# xattention: stride=8, threshold=0.9 (config defaults).
#
# Two models run in PARALLEL: Qwen3 on GPUs 0-3, Llama on GPUs 4-7, 4 workers each.
# Within each model: for each length -> reuse_v1 then xattn (serial). Separate
# work_dirs per (model, method, length) so nothing clobbers.
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export RULER_NUM_SAMPLES=200
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-layer selected-block logging for reuse_v1 (xattn ignores it).
export REUSE_V1_LOG_BLOCKS=1

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
cd "${REPO_ROOT}/eval/benchmarks"

QWEN3_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt"
LLAMA_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt"
for L in "${QWEN3_LABEL}" "${LLAMA_LABEL}"; do
  [ -f "${L}" ] || { echo "ERROR: label not found: ${L}" >&2; exit 1; }
done

LENGTHS=(8 16 32 64 128)
LOGDIR="${REPO_ROOT}/logs/ruler_matrix"
mkdir -p "${LOGDIR}"

run_qwen3() {
  local gpus="0,1,2,3"
  for K in "${LENGTHS[@]}"; do
    echo "### [qwen3] reuse_v1 total lastfull @ ${K}k"
    RULER_MAX_SEQ_LEN_K="${K}" CUDA_VISIBLE_DEVICES="${gpus}" \
      QWEN3_REUSE_V1_LABEL_PATH="${QWEN3_LABEL}" QWEN3_REUSE_V1_TOP_P=0.9 \
      QWEN3_REUSE_V1_LAST_Q_FULL=1 \
      "${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 4 \
        -w "./results/ruler/exps_qwen3_8b/reuse_v1_total_lastfull_${K}k"
    echo "### [qwen3] reuse_v1 @ ${K}k exit=$?"

    echo "### [qwen3] xattn @ ${K}k"
    RULER_MAX_SEQ_LEN_K="${K}" CUDA_VISIBLE_DEVICES="${gpus}" \
      "${OC}" ruler/exps_qwen3_8b/xattn.py --max-num-workers 4 \
        -w "./results/ruler/exps_qwen3_8b/xattn_${K}k"
    echo "### [qwen3] xattn @ ${K}k exit=$?"
  done
}

run_llama() {
  local gpus="4,5,6,7"
  for K in "${LENGTHS[@]}"; do
    echo "### [llama] reuse_v1 total lastfull @ ${K}k"
    RULER_MAX_SEQ_LEN_K="${K}" CUDA_VISIBLE_DEVICES="${gpus}" \
      LLAMA_REUSE_V1_LABEL_PATH="${LLAMA_LABEL}" LLAMA_REUSE_V1_TOP_P=0.9 \
      LLAMA_REUSE_V1_LAST_Q_FULL=1 \
      "${OC}" ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 4 \
        -w "./results/ruler/exps_llama_31_8b/reuse_v1_total_lastfull_${K}k"
    echo "### [llama] reuse_v1 @ ${K}k exit=$?"

    echo "### [llama] xattn @ ${K}k"
    RULER_MAX_SEQ_LEN_K="${K}" CUDA_VISIBLE_DEVICES="${gpus}" \
      "${OC}" ruler/exps_llama_31_8b/xattn.py --max-num-workers 4 \
        -w "./results/ruler/exps_llama_31_8b/xattn_${K}k"
    echo "### [llama] xattn @ ${K}k exit=$?"
  done
}

run_qwen3 > "${LOGDIR}/qwen3.log" 2>&1 &
PID_Q=$!
run_llama > "${LOGDIR}/llama.log" 2>&1 &
PID_L=$!
echo "qwen3 pid=${PID_Q} (GPUs 0-3, log ${LOGDIR}/qwen3.log)"
echo "llama pid=${PID_L} (GPUs 4-7, log ${LOGDIR}/llama.log)"
wait "${PID_Q}"; echo "=== qwen3 stream done exit=$? ==="
wait "${PID_L}"; echo "=== llama stream done exit=$? ==="
echo "=== ALL DONE ==="
