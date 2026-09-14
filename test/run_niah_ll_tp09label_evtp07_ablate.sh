#!/bin/bash
# NIAH topp=0.7 ablation for the Llama-3.1-8B tp=0.9-trained mb8/xb64 label.
# Extends the earlier evalTP 0.8/0.9 sweep with evalTP=0.7 at both eval
# MAX_BLOCKS in {64, 1024}. min_blocks stays 8. 2 sequential 8-GPU sweeps.
# Tag encodes training xb64 vs eval xb explicitly (trainXB64 / evalXB<N>).
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
LOGDIR="${REPO_ROOT}/test/results-912/niah_ll_tp09label_evtp07_ablate_logs"
mkdir -p "${LOGDIR}"

LL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"

run () {
  local evtp="$1" maxblk="$2" name="$3"
  echo "=== [${name}] START $(date -Is) ==="
  MODEL="${LL}" LABEL="${LL_LABEL}" \
  TOP_P="${evtp}" MIN_BLOCKS=8 MAX_BLOCKS="${maxblk}" SELECT_MODE=topp METHOD=reuse_v1 \
  RUN_TAG="trainTP0.9_evalTP${evtp}_rw0.003_sp0.8_mb8_trainXB64_evalXB${maxblk}" \
    bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/${name}.log" 2>&1
  echo "=== [${name}] EXIT $? $(date -Is) ==="
}

run 0.7 64   "ll_evtp0.7_xb64"
run 0.7 1024 "ll_evtp0.7_xb1024"
echo "=== ALL Llama tp0.9-label evalTP0.7 xb ablation DONE $(date -Is) ==="
