#!/bin/bash
# NIAH fixed topk=64 test for the tp=0.9 mb8/xb64 labels (Qwen3 + Llama-3.1-8B).
# SELECT_MODE=topk + BUDGET=64: each q-block selects exactly 64 content blocks,
# so eval xb (max_blocks) / min_blocks are irrelevant -> single run per label.
# RUN_TAG encodes evalTOPK64.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
LOGDIR="${REPO_ROOT}/test/results-912/niah_tp09label_evtopk64_logs"
mkdir -p "${LOGDIR}"

QW="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
QW_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"

run () {
  local model="$1" label="$2" name="$3"
  echo "=== [${name}] START $(date -Is) ==="
  MODEL="${model}" LABEL="${label}" \
  SELECT_MODE=topk BUDGET=64 MIN_BLOCKS=8 MAX_BLOCKS=1024 METHOD=reuse_v1 \
  RUN_TAG="trainTP0.9_evalTOPK64_rw0.003_sp0.8_mb8_trainXB64" \
    bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/${name}.log" 2>&1
  echo "=== [${name}] EXIT $? $(date -Is) ==="
}

run "${QW}" "${QW_LABEL}" "qwen3_topk64"
run "${LL}" "${LL_LABEL}" "llama_topk64"
echo "=== ALL tp0.9-label topk=64 NIAH DONE $(date -Is) ==="
