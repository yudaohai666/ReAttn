#!/bin/bash
# reg_weight sweep: rw=0.003 / 0.004 / 0.005 labels (all trained with topp/top_p=0.7),
# each evaluated at inference top_p 0.7 (matched) and 0.9 (mismatched), sequentially.
set -uo pipefail

REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
BASE="${REPO_ROOT}/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"

for rw in 0.003 0.004 0.005; do
  LBL="${BASE}/hc-orig-rw=${rw}-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt"
  for tp in 0.7 0.9; do
    echo "############ rw=${rw} eval top_p=${tp} ############"
    LABEL="${LBL}" TOP_P="${tp}" RUN_TAG="rw${rw}_labeltp0.7_eval${tp}" LAST_Q_FULL=0 \
      bash "${REPO_ROOT}/test/run_niah_pre_topp.sh"
  done
done
