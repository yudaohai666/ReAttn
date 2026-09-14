#!/bin/bash
# Cross-check: eval the two tp=0.8-trained mb8/xb64 labels with MISMATCHED
# TOP_P=0.9 (training used top_p=0.8). Measures top_p transfer sensitivity.
# Distinct RUN_TAG so it does not overwrite the matched-TOP_P=0.8 results.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/test/results—910/niah_tp08label_eval09_mb8_xb64_logs"
mkdir -p "${LOGDIR}"

export TOP_P=0.9 MIN_BLOCKS=8 MAX_BLOCKS=64 SELECT_MODE=topp METHOD=reuse_v1

QW_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"

echo "=== [qwen3] START $(date -Is) ==="
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B" \
LABEL="${QW_LABEL}" RUN_TAG="trainTP0.8_evalTP0.9_rw0.003_sp0.8_mb8_xb64" \
  bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/qwen3.log" 2>&1
echo "=== [qwen3] EXIT $? $(date -Is) ==="

echo "=== [llama] START $(date -Is) ==="
MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct" \
LABEL="${LL_LABEL}" RUN_TAG="trainTP0.8_evalTP0.9_rw0.003_sp0.8_mb8_xb64" \
  bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/llama.log" 2>&1
echo "=== [llama] EXIT $? $(date -Is) ==="
echo "=== ALL trainTP0.8/evalTP0.9 NIAH DONE $(date -Is) ==="
