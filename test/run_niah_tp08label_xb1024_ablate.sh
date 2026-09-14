#!/bin/bash
# Ablation: the two tp=0.8-trained mb8 labels, evaluated with MAX_BLOCKS=1024
# (xb1024) under two inference TOP_P values (0.9 and 0.8). min_blocks stays 8.
# 4 sequential 8-GPU sweeps. Distinct RUN_TAGs per (label, eval_top_p).
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/test/results—910/niah_tp08label_xb1024_ablate_logs"
mkdir -p "${LOGDIR}"

QW="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
LL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
QW_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"

run () {
  local model="$1" label="$2" evtp="$3" name="$4"
  echo "=== [${name}] START $(date -Is) ==="
  MODEL="${model}" LABEL="${label}" \
  TOP_P="${evtp}" MIN_BLOCKS=8 MAX_BLOCKS=1024 SELECT_MODE=topp METHOD=reuse_v1 \
  RUN_TAG="trainTP0.8_evalTP${evtp}_rw0.003_sp0.8_mb8_xb1024" \
    bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/${name}.log" 2>&1
  echo "=== [${name}] EXIT $? $(date -Is) ==="
}

run "${QW}" "${QW_LABEL}" 0.9 "qwen3_evtp0.9"
run "${QW}" "${QW_LABEL}" 0.8 "qwen3_evtp0.8"
run "${LL}" "${LL_LABEL}" 0.9 "llama_evtp0.9"
run "${LL}" "${LL_LABEL}" 0.8 "llama_evtp0.8"
echo "=== ALL tp0.8-label xb1024 ablation DONE $(date -Is) ==="
