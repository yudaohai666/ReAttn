export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Checkpoint (label) path used by the sparse_reuse models. Only read by the
# *_sparse_reuse configs; harmless for other patch types.
export LLAMA_SPARSE_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn/ckp/Llama-3.1-8B-Instruct/full_head_new_02.pt
export LLAMA_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn_h/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8.pt
# export QWEN_SPARSE_LABEL_PATH=/root/paddlejob/inference-public/yudaohai/sparse_attention/pbs-attn/ckp/Qwen2.5-7B-Instruct-1M/full_head.pt

cd eval/benchmarks

########################################################
# Llama 3.1 8B Instruct
########################################################
# PBS-Attn
opencompass longbenchv2/exps_llama_31_8b/pbs.py --max-num-workers 8

# Sparse-Reuse (our method)
# opencompass longbenchv2/exps_llama_31_8b/sparse_reuse.py --max-num-workers 8
# Reuse-v1
# opencompass longbenchv2/exps_llama_31_8b/reuse_v1.py --max-num-workers 8

# # Flash Attention
opencompass longbenchv2/exps_llama_31_8b/flashattn.py --max-num-workers 8
# # Minference
# opencompass longbenchv2/exps_llama_31_8b/minference.py --max-num-workers 8
# # FlexPrefill
# opencompass longbenchv2/exps_llama_31_8b/flexprefill.py --max-num-workers 8
# # XAttention
opencompass longbenchv2/exps_llama_31_8b/xattn.py --max-num-workers 8
# # MeanPooling
# opencompass longbenchv2/exps_llama_31_8b/meanpooling.py --max-num-workers 8

# ########################################################
# # Qwen 2.5 7B 1M (TP=4 since there are only 4 kv heads)
# ########################################################
# # PBS-Attn
# opencompass longbenchv2/exps_qwen_25_7b_1m/pbs.py --max-num-workers 4

# # Sparse-Reuse (our method) -- single GPU (no TP support)
# opencompass longbenchv2/exps_qwen25_7b_1m/sparse_reuse.py --max-num-workers 1

# # Flash Attention
# opencompass longbenchv2/exps_qwen_25_7b_1m/flashattn.py --max-num-workers 4
# # Minference
# opencompass longbenchv2/exps_qwen_25_7b_1m/minference.py --max-num-workers 4
# # FlexPrefill
# opencompass longbenchv2/exps_qwen_25_7b_1m/flexprefill.py --max-num-workers 4
# # XAttention
# opencompass longbenchv2/exps_qwen_25_7b_1m/xattn.py --max-num-workers 4
# # MeanPooling
# opencompass longbenchv2 /exps_qwen_25_7b_1m/meanpooling.py --max-num-workers 4