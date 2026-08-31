# export CUDA_VISIBLE_DEVICES=7

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT/eval/efficiency"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Checkpoint (label) path used by the sparse_reuse method.
export LLAMA_SPARSE_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn/ckp/Llama-3.1-8B-Instruct/full_head_new_02.pt

# Checkpoint (label) path used by the reuse_v1 method.
export LLAMA_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn_h/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8.pt

# # LENGTHS=(128)
# LENGTHS=(64 32 16 8)

# for length in "${LENGTHS[@]}"; do
#     echo "Evaluating sparse_reuse at ${length}K"
#     CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
#         --method sparse_reuse \
#         --len $((length * 1024)) \
#         --model_name /root/paddlejob/inference-public/yudaohai/data/Llama-3.1-8B-Instruct \
#         --label_path "$LLAMA_SPARSE_LABEL_PATH" \
#         --sparse_phase prefill \
#         --sink_blocks 1 \
#         --local_blocks 2 \
#         --topk 32
# done

LENGTHS=(128)
# LENGTHS=(64 32 16 8)

# for length in "${LENGTHS[@]}"; do
#     echo "Evaluating reuse_v1 at ${length}K"
#     CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
#         --method reuse_v1 \
#         --len $((length * 1024)) \
#         --model_name /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct \
#         --label_path "$LLAMA_REUSE_V1_LABEL_PATH" \
#         --budget 32 \
#         --sink_blocks 1 \
#         --local_blocks 2 \
#         --select_mode topp \
#         --top_p 0.9 \
#         --min_blocks 8 \
#         --max_blocks 64
# done


for length in "${LENGTHS[@]}"; do
    echo "Evaluating reuse_v1 at ${length}K"
    CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
        --method reuse_v1 \
        --len $((length * 1024)) \
        --model_name /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct \
        --label_path "$LLAMA_REUSE_V1_LABEL_PATH" \
        --budget 32 \
        --sink_blocks 1 \
        --local_blocks 2 \
        --select_mode topk \
        --topk_ratio 0.05 \
        --last_q_full True
done

LENGTHS=(128)

# for length in "${LENGTHS[@]}"; do
#     echo "Evaluating reuse_v1 at ${length}K"
#     CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
#         --method reuse_v1 \
#         --len $((length * 1024)) \
#         --model_name /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct \
#         --label_path "$LLAMA_REUSE_V1_LABEL_PATH" \
#         --budget 32 \
#         --sink_blocks 1 \
#         --local_blocks 2 \
#         --select_mode topp \
#         --top_p 0.9 \
#         --min_blocks 8 \
#         --max_blocks 64 \
#         --last_q_full True
# done

# Evaluate all methods on 8k, 16k, 32k, 64k, 128k
METHODS=(
    pbs
    # flashattn
    # minference
    # flexprefill
    # xattention
#     meanpooling
)

# LENGTHS=(8 16 32 64)
# LENGTHS=(512 256 128 64 32 16 8)
LENGTHS=(128)



for method in "${METHODS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "Evaluating method=${method}, length=${length}k"

        CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
            --method "$method" \
            --len $((length * 1024))
    done
done



# METHOD="pbs" # baselines: flashattn, minference, flexprefill, xattention, meanpooling


# # Evaluate on 8k, 16k, 32k, 64k, 128k
# for len in 8 16 32 64 128; do
#      CUDA_VISIBLE_DEVICES=0 python eval_efficiency.py \
#           --method $METHOD \
#           --len $((len * 1024))
# done

# # Evaluate on 256k, tp=4
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone eval_efficiency.py \
#      --method $METHOD \
#      --len $((256 * 1024))


# # Evaluate on 512k, tp=8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --standalone eval_efficiency.py \
#      --method $METHOD \
#      --len $((512 * 1024))
