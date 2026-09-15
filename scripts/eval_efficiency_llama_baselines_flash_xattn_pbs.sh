#!/bin/bash
# Efficiency (prefill latency) baselines on Llama-3.1-8B-Instruct, same length
# sweep as the "M" reuse_v1 run: 8k/16k/32k/64k/128k, single GPU.
# Baselines: flashattn (dense), xattention, pbs. Results tagged per method.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/eval/efficiency"

PY="${REPO_ROOT}/.venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL="/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
OUTDIR="${REPO_ROOT}/eval/efficiency/results/M_llama_baselines"

LENGTHS=(8 16 32 64 128)
METHODS=(flashattn xattention pbs)

for method in "${METHODS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    echo "=== [efficiency ${method} Llama ${length}k] START $(date -Is) ==="
    CUDA_VISIBLE_DEVICES=0 "$PY" eval_efficiency.py \
        --method "$method" \
        --len $((length * 1024)) \
        --model_name "$MODEL" \
        --output_dir "$OUTDIR"
    echo "=== [efficiency ${method} Llama ${length}k] EXIT $? $(date -Is) ==="
  done
done
echo "=== ALL efficiency baselines (flashattn/xattention/pbs) Llama 8k..128k DONE $(date -Is) ==="
