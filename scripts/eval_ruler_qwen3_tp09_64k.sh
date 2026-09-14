#!/bin/bash
# RULER 64k for the Qwen3 tp=0.9 mb8/xb64 trained label, evaluated with
# inference TOP_P=0.9 and xb (max_blocks)=1024. last_q_full=1 (matches the
# lastfull driver convention). Single 64k context point, all 5 RULER task
# groups, 8 parallel GPU workers.
set -uo pipefail
REPO_ROOT="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
OC="${REPO_ROOT}/.venv/bin/opencompass"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
cd "${REPO_ROOT}/eval/benchmarks"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# RULER cwe/fwe/vt build contexts with tiktoken (cl100k_base). This node has no
# internet egress, so point tiktoken at the pre-downloaded blob cache.
export TIKTOKEN_CACHE_DIR="${REPO_ROOT}/.cache/tiktoken"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export QWEN3_REUSE_V1_LABEL_PATH="${REPO_ROOT}/attn_patterns/reuse_v1/Qwen3-8B/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
export QWEN3_REUSE_V1_TOP_P=0.9
export QWEN3_REUSE_V1_MAX_BLOCKS=1024
export QWEN3_REUSE_V1_MIN_BLOCKS=8
export QWEN3_REUSE_V1_LAST_Q_FULL="${QWEN3_REUSE_V1_LAST_Q_FULL:-1}"

export RULER_MAX_SEQ_LEN_K="${RULER_MAX_SEQ_LEN_K:-64}"
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-200}"

echo "=== [qwen3 reuse_v1 RULER ${RULER_MAX_SEQ_LEN_K}k] START $(date -Is) ==="
echo "  label = ${QWEN3_REUSE_V1_LABEL_PATH}"
echo "  top_p = ${QWEN3_REUSE_V1_TOP_P}  max_blocks = ${QWEN3_REUSE_V1_MAX_BLOCKS}  last_q_full = ${QWEN3_REUSE_V1_LAST_Q_FULL}"
"${OC}" ruler/exps_qwen3_8b/reuse_v1.py --max-num-workers 8
echo "=== [qwen3 reuse_v1 RULER ${RULER_MAX_SEQ_LEN_K}k] EXIT $? $(date -Is) ==="
