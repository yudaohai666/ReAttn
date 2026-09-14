#!/bin/bash
# NIAH ablation for the Llama-3.1-8B tp=0.8-trained mb8/xb64 label.
# 2x2 grid: MAX_BLOCKS in {64, 1024} (xb64/xb1024) x eval TOP_P in {0.9, 0.8}.
# min_blocks stays 8. 4 sequential 8-GPU sweeps, each with a distinct RUN_TAG.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
LOGDIR="${REPO_ROOT}/test/results-912/niah_ll_tp08label_xb_topp_ablate_logs"
mkdir -p "${LOGDIR}"

LL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"

run () {
  local evtp="$1" maxblk="$2" name="$3"
  echo "=== [${name}] START $(date -Is) ==="
  MODEL="${LL}" LABEL="${LL_LABEL}" \
  TOP_P="${evtp}" MIN_BLOCKS=8 MAX_BLOCKS="${maxblk}" SELECT_MODE=topp METHOD=reuse_v1 \
  RUN_TAG="trainTP0.8_evalTP${evtp}_rw0.003_sp0.8_mb8_trainXB64_evalXB${maxblk}" \
    bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/${name}.log" 2>&1
  echo "=== [${name}] EXIT $? $(date -Is) ==="
}

run 0.9 64   "ll_evtp0.9_xb64"
run 0.8 64   "ll_evtp0.8_xb64"
run 0.9 1024 "ll_evtp0.9_xb1024"
run 0.8 1024 "ll_evtp0.8_xb1024"
echo "=== ALL Llama tp0.8-label xb/topp ablation DONE $(date -Is) ==="
