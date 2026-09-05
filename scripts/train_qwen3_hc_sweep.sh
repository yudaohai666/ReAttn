#!/usr/bin/env bash
# Sequential sweep over reg_weight for Qwen3-8B HC training.
# reg_weight: 0.003, 0.004, 0.005, 0.006, 0.007
# Other hyperparams fixed from original Llama run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B
VENV="${REPO_ROOT}/.venv/bin/activate"

# Fixed params
CTX_MIN=8000
CTX_MAX=128000
LR=0.01
NUM_PASSKEY=10
SP_SIZE=8
INITIAL_VALUE=0.0
TARGET_SPARSITY=0.8
TOP_P=0.7

source "${VENV}"
cd "${REPO_ROOT}"

for RW in 0.003 0.004 0.005 0.006 0.007; do
    echo "========================================"
    echo "Starting run: reg_weight=${RW}"
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
