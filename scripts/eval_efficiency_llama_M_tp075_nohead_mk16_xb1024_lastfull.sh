#!/bin/bash
# Efficiency (prefill latency) for the "M" reuse_v1 config on Llama-3.1-8B-Instruct.
# M = tp0.75 nohead (per_head_topp=False) min_blocks=16 xb(max_blocks)=1024 last_q_full=1.
# Uses the Llama tp=0.9 mb8/xb64 trained label (same sparsity source as RULER Req I).
# Sweeps 8k/16k/32k/64k/128k, single GPU, results tagged in its own output_dir.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/eval/efficiency"

# Use the repo venv (py3.10) that the RULER runs use, not the system python (3.12).
PY="${REPO_ROOT}/.venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Offline node: LongBench-v2 is pre-cached; avoid HEAD requests to the hub loader.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL="${REPO_ROOT}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
OUTDIR="${REPO_ROOT}/eval/efficiency/results/M_llama_tp075_nohead_mk16_xb1024_lastfull"

LENGTHS=(8 16 32 64 128)

for length in "${LENGTHS[@]}"; do
    echo "=== [efficiency reuse_v1 M-config Llama ${length}k] START $(date -Is) ==="
    echo "  top_p=0.75 per_head_topp=False min_blocks=16 max_blocks=1024 last_q_full=True"
    CUDA_VISIBLE_DEVICES=0 "$PY" eval_efficiency.py \
        --method reuse_v1 \
        --len $((length * 1024)) \
        --model_name "$MODEL" \
        --label_path "$LABEL" \
        --budget 32 \
        --sink_blocks 1 \
        --local_blocks 2 \
        --select_mode topp \
        --top_p 0.75 \
        --min_blocks 16 \
        --max_blocks 1024 \
        --per_head_topp False \
        --last_q_full True \
        --output_dir "$OUTDIR"
    echo "=== [efficiency reuse_v1 M-config Llama ${length}k] EXIT $? $(date -Is) ==="
done
echo "=== ALL efficiency reuse_v1 M-config Llama 8k+16k+32k+64k+128k DONE $(date -Is) ==="
