#!/bin/bash
# Same label (rw=0.002, trained with topp/top_p=0.7), evaluated at inference
# top_p 0.7 (matched) and 0.9 (mismatched), sequentially.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
RUN_NAME="hc-orig-rw=0.002-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_128000-multi_passkey10-sp8"
LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct/${RUN_NAME}/label.pt"

for tp in 0.7 0.9; do
  echo "############ eval top_p=${tp} ############"
  LABEL="${LABEL}" TOP_P="${tp}" RUN_TAG="rw0.002_labeltp0.7_eval${tp}" LAST_Q_FULL=0 \
    bash "${REPO_ROOT}/test/run_niah_pre_topp.sh"
done
