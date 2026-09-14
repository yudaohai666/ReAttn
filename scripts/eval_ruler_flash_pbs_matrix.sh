#!/bin/bash
# RULER matrix part 2: flashattn (dense baseline) + pbs, for
# Qwen3-8B + Llama-3.1-8B-Instruct, lengths {8k,16k,32k,64k,128k}, 200 samples.
# Neither method uses a reuse label / topp / lastfull. Two models run in PARALLEL
# (Qwen3 GPUs 0-3, Llama GPUs 4-7, 4 workers each); within each model, per length:
# flashattn then pbs. Separate work_dirs per (model, method, length).
set -uo pipefail

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export RULER_NUM_SAMPLES=200
export TIKTOKEN_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/.cache/tiktoken
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
cd "${REPO_ROOT}/eval/benchmarks"

LENGTHS=(8 16 32 64 128)
LOGDIR="${REPO_ROOT}/logs/ruler_matrix"
mkdir -p "${LOGDIR}"

run_qwen3() {
  local gpus="0,1,2,3"
  for K in "${LENGTHS[@]}"; do
    for M in flashattn pbs; do
      echo "### [qwen3] ${M} @ ${K}k"
      RULER_MAX_SEQ_LEN_K="${K}" CUDA_VISIBLE_DEVICES="${gpus}" \
        "${OC}" ruler/exps_qwen3_8b/${M}.py --max-num-workers 4 \
          -w "./results/ruler/exps_qwen3_8b/${M}_${K}k"
      echo "### [qwen3] ${M} @ ${K}k exit=$?"
    done
  done
}

run_llama() {
  local gpus="4,5,6,7"
  for K in "${LENGTHS[@]}"; do
    for M in flashattn pbs; do
      echo "### [llama] ${M} @ ${K}k"
      RULER_MAX_SEQ_LEN_K="${K}" CUDA_VISIBLE_DEVICES="${gpus}" \
        "${OC}" ruler/exps_llama_31_8b/${M}.py --max-num-workers 4 \
          -w "./results/ruler/exps_llama_31_8b/${M}_${K}k"
      echo "### [llama] ${M} @ ${K}k exit=$?"
    done
  done
}

run_qwen3 > "${LOGDIR}/qwen3_flashpbs.log" 2>&1 &
PID_Q=$!
run_llama > "${LOGDIR}/llama_flashpbs.log" 2>&1 &
PID_L=$!
echo "qwen3 flashpbs pid=${PID_Q} (GPUs 0-3)"
echo "llama flashpbs pid=${PID_L} (GPUs 4-7)"
wait "${PID_Q}"; echo "=== qwen3 flashpbs done exit=$? ==="
wait "${PID_L}"; echo "=== llama flashpbs done exit=$? ==="
echo "=== FLASH+PBS ALL DONE ==="
