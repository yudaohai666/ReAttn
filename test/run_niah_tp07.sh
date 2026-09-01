#!/bin/bash
# NIAH for the post-fix reuse_v1 label trained with select_mode=topp, top_p=0.7,
# reg_weight=0.0013, target_sparsity=0.8. Eval top_p matches the trained top_p.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
RUN_NAME="hc-orig-rw=0.0013-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_128000-multi_passkey10-sp8"

export LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct/${RUN_NAME}/label.pt"
export TOP_P=0.7
export RUN_TAG="topp0.7_rw0.0013_sp0.8"
export LAST_Q_FULL=0

exec bash "${REPO_ROOT}/test/run_niah_pre_topp.sh"
