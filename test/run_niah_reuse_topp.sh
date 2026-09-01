#!/bin/bash
# NIAH validation: reuse_v1 + topp(0.9) block selection,
# label = hc-orig-rw=0.0013-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8
set -euo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
LABEL_ROOT="${REPO_ROOT}/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL_DIR="hc-orig-rw=0.0013-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8"

export LABEL_PATH="${LABEL_ROOT}/${LABEL_DIR}/label.pt"
[ -f "${LABEL_PATH}" ] || { echo "label not found: ${LABEL_PATH}" >&2; exit 1; }

export METHOD=reuse_v1
export SELECT_MODE=topp
export TOP_P=0.9
export MIN_BLOCKS=8
export MAX_BLOCKS=64
export BLOCK_SIZE=128
export SEGMENT_SIZE=2048
export SINK_BLOCKS=1
export LOCAL_BLOCKS=2
export ATTN_IMPL=sdpa
export GPU="${GPU:-0}"
export RUN_TAG="${RUN_TAG:-topp0.9_rw0.0013_sp0.8}"

# Allow a quick smoke test: SMOKE=1 bash run_niah_reuse_topp.sh
if [ "${SMOKE:-0}" = "1" ]; then
  export LENGTHS="16384"
  export DEPTHS="50"
  export RUN_TAG="${RUN_TAG}_smoke"
fi

bash "${REPO_ROOT}/test/run_niah.sh"
