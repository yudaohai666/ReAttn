#!/usr/bin/env bash
# Sweep: reg_weight=0.002,0.0025,0.003, top_p=0.9, Qwen3-8B
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B
VENV="${REPO_ROOT}/.venv/bin/activate"

CTX_MIN=8000
CTX_MAX=128000
LR=0.01
NUM_PASSKEY=10
SP_SIZE=8
INITIAL_VALUE=0.0
TARGET_SPARSITY=0.8
TOP_P=0.9

source "${VENV}"
cd "${REPO_ROOT}"

for RW in 0.002 0.0025 0.003; do
    echo "========================================"
    echo "Starting run: reg_weight=${RW} top_p=${TOP_P}"
    echo "========================================"
    bash scripts/train_reuse_hc.sh \
        "${MODEL}" \
        "${CTX_MIN}" "${CTX_MAX}" \
        "${LR}" "${NUM_PASSKEY}" "${SP_SIZE}" \
        "${RW}" "${INITIAL_VALUE}" "${TARGET_SPARSITY}" "${TOP_P}"
    echo "========================================"
    echo "Finished run: reg_weight=${RW}"
    echo "========================================"
done

echo "All sweeps done."
