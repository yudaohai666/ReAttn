#!/usr/bin/env bash
# Retrain both reuse_v1 labels with the corrected topp (total-mass norm),
# min_blocks=4, max_blocks=1024 (effectively uncapped). Sequential: each run
# needs all 8 GPUs at sp_size=8. Matches the -912 reference config otherwise:
#   reg_weight=0.003 init=0.0 target_sparsity=0.8 top_p=0.9 lr=0.01
#   ctx=8000_128000 num_passkey=10 sp_size=8 num_steps=2000
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
cd "${REPO_ROOT}"

DATA="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data"
LLAMA="${DATA}/Llama-3.1-8B-Instruct"
QWEN3="${DATA}/Qwen3-8B"

LOGDIR="${REPO_ROOT}/logs/retrain_topp_mb4_xb1024"
mkdir -p "${LOGDIR}"

# positional: model ctx_min ctx_max lr num_passkey sp reg init sparsity top_p min_blocks max_blocks
run_one() {
  local model="$1" tag="$2"
  echo "=== [${tag}] START $(date -Is) ==="
  bash scripts/train_reuse_hc.sh \
    "${model}" 8000 128000 0.01 10 8 0.003 0.0 0.8 0.9 4 1024 \
    > "${LOGDIR}/${tag}.log" 2>&1
  echo "=== [${tag}] EXIT $? $(date -Is) ==="
}

run_one "${LLAMA}" "llama"
run_one "${QWEN3}" "qwen3"
echo "=== ALL RETRAIN DONE $(date -Is) ==="
