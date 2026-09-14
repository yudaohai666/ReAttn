#!/bin/bash
# Ablation for two more labels, same grid as before: eval TOP_P in {0.9,0.8} x
# eval xb in {64,1024}, MIN_BLOCKS=8 fixed.
#   A) Qwen3  label trained tp=0.9 mb8/xb64
#   B) Llama  label trained tp=0.9 mb4/xb1024
# Qwen3 (evalTP0.9,xb64) is skipped -- already have it as
#   Qwen3-8B_reuse_v1_topp0.9_rw0.003_sp0.8_mb8_xb64 (avg 83.92 / strict 75.6%).
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"
LOGDIR="${REPO_ROOT}/test/results—910/niah_ablate2_logs"
mkdir -p "${LOGDIR}"

QW="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B"
LL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
QW_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
LL_LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb4-xb1024/label.pt"

run () {
  local model="$1" label="$2" evtp="$3" xb="$4" tag="$5" name="$6"
  echo "=== [${name}] START $(date -Is) ==="
  MODEL="${model}" LABEL="${label}" \
  TOP_P="${evtp}" MIN_BLOCKS=8 MAX_BLOCKS="${xb}" SELECT_MODE=topp METHOD=reuse_v1 \
  RUN_TAG="${tag}" \
    bash "${REPO_ROOT}/test/run_niah_reuse_topp_par.sh" > "${LOGDIR}/${name}.log" 2>&1
  echo "=== [${name}] EXIT $? $(date -Is) ==="
}

# A) Qwen3 tp=0.9 mb8/xb64 label -- 3 remaining cells
run "${QW}" "${QW_LABEL}" 0.8 64   "qwlbl_TP0.9mb8xb64_evalTP0.8_xb64"   "qw_evTP0.8_xb64"
run "${QW}" "${QW_LABEL}" 0.9 1024 "qwlbl_TP0.9mb8xb64_evalTP0.9_xb1024" "qw_evTP0.9_xb1024"
run "${QW}" "${QW_LABEL}" 0.8 1024 "qwlbl_TP0.9mb8xb64_evalTP0.8_xb1024" "qw_evTP0.8_xb1024"

# B) Llama tp=0.9 mb4/xb1024 label -- full 4-cell grid
run "${LL}" "${LL_LABEL}" 0.9 64   "lllbl_TP0.9mb4xb1024_evalTP0.9_xb64"   "ll_evTP0.9_xb64"
run "${LL}" "${LL_LABEL}" 0.8 64   "lllbl_TP0.9mb4xb1024_evalTP0.8_xb64"   "ll_evTP0.8_xb64"
run "${LL}" "${LL_LABEL}" 0.9 1024 "lllbl_TP0.9mb4xb1024_evalTP0.9_xb1024" "ll_evTP0.9_xb1024"
run "${LL}" "${LL_LABEL}" 0.8 1024 "lllbl_TP0.9mb4xb1024_evalTP0.8_xb1024" "ll_evTP0.8_xb1024"

echo "=== ALL ablate2 DONE $(date -Is) ==="
