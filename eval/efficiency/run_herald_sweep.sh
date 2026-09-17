#!/bin/bash
# reuse_v1 fused=1, config: evtp0.95 / nohead / minblk8 / xb48 / lastfull
# length sweep 8/16/32/64/128k on a single GPU, real LongBench-v2 harness.
set -u
REPO_ROOT=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn
cd "$REPO_ROOT"
source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export REUSE_V1_FUSED_ANCHOR=1
export CUDA_VISIBLE_DEVICES=0

LABEL="$REPO_ROOT/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
MODEL=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct
OUT="$REPO_ROOT/eval/efficiency/results_herald_evtp095_minblk8_xb48_lastfull_fused"

cd "$REPO_ROOT/eval/efficiency"
for length in 8 16 32 64 128; do
    echo "############ reuse_v1 fused=1 length=${length}k ############"
    python eval_efficiency.py \
        --method reuse_v1 \
        --len $((length * 1024)) \
        --n_examples 3 \
        --num_warmup_iter 3 \
        --output_dir "$OUT" \
        --model_name "$MODEL" \
        --label_path "$LABEL" \
        --budget 32 \
        --sink_blocks 1 \
        --local_blocks 2 \
        --select_mode topp \
        --top_p 0.95 \
        --min_blocks 8 \
        --max_blocks 48 \
        --last_q_full True
done
echo "############ ALL DONE ############"
