export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RULER_NUM_SAMPLES=200
# export RULER_TASKS=niah
# export RULER_SUBTASKS=niah_single_1

# Checkpoint (label) path used by the sparse_reuse models. Only read by the
# *_sparse_reuse configs; harmless for other patch types.
export LLAMA_SPARSE_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn/ckp/Llama-3.1-8B-Instruct/full_head_new_02.pt
export LLAMA_REUSE_V1_LABEL_PATH=/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn_h/attn_patterns/reuse_v1/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/hc-orig-rw=0.0013-init=0.0-sp=0.8-lr=0.01-ctx=8000_128000-multi_passkey10-sp8.pt
# export QWEN_SPARSE_LABEL_PATH=/root/paddlejob/inference-public/yudaohai/sparse_attention/pbs-attn/ckp/Qwen2.5-7B-Instruct-1M/full_head.pt

cd eval/benchmarks

# RULER. Defaults to 128k / 100 samples / all sub-tasks. Tune at runtime via
# env vars (no need to edit the configs):
#   RULER_NUM_SAMPLES   samples per sub-task            (default 100)
#   RULER_MAX_SEQ_LEN_K context length in K tokens      (default 128; use 4/8/16/32/64/128...)
#   RULER_TASKS         subset of cwe,fwe,niah,qa,vt    (default all)
#   RULER_SUBTASKS      finer abbr-substring filter     (default empty = keep all)
#                       e.g. niah_single_1 keeps only ruler_niah_single_1
#   RULER_TOKENIZER     tokenizer controlling ctx len   (default gpt-4)
# Quick smoke test (few samples, short ctx, single sub-task, ~minutes):
#   RULER_NUM_SAMPLES=5 RULER_MAX_SEQ_LEN_K=8 RULER_TASKS=niah RULER_SUBTASKS=niah_single_1 \
#     opencompass ruler/exps_llama_31_8b/flashattn.py --max-num-workers 1 --debug

########################################################
# Llama 3.1 8B Instruct
########################################################
# PBS-Attn
# opencompass ruler/exps_llama_31_8b/pbs.py --max-num-workers 8
# Sparse-Reuse (our method)
# opencompass ruler/exps_llama_31_8b/sparse_reuse.py --max-num-workers 8
# Flash Attention
# opencompass ruler/exps_llama_31_8b/flashattn.py --max-num-workers 8
# Minference
# opencompass ruler/exps_llama_31_8b/minference.py --max-num-workers 8
# FlexPrefill
# opencompass ruler/exps_llama_31_8b/flexprefill.py --max-num-workers 8
# XAttention
# RULER_MAX_SEQ_LEN_K=8 opencompass ruler/exps_llama_31_8b/xattn.py --max-num-workers 8

# RULER_MAX_SEQ_LEN_K=16 opencompass ruler/exps_llama_31_8b/xattn.py --max-num-workers 8

# RULER_MAX_SEQ_LEN_K=32 opencompass ruler/exps_llama_31_8b/xattn.py --max-num-workers 8

# RULER_MAX_SEQ_LEN_K=64 opencompass ruler/exps_llama_31_8b/xattn.py --max-num-workers 8

# RULER_MAX_SEQ_LEN_K=128 opencompass ruler/exps_llama_31_8b/xattn.py --max-num-workers 8

# Reuse-v1
RULER_MAX_SEQ_LEN_K=8 opencompass ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8

RULER_MAX_SEQ_LEN_K=16 opencompass ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8

RULER_MAX_SEQ_LEN_K=32 opencompass ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8

RULER_MAX_SEQ_LEN_K=64 opencompass ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8

RULER_MAX_SEQ_LEN_K=128 opencompass ruler/exps_llama_31_8b/reuse_v1.py --max-num-workers 8

# MeanPooling
# opencompass ruler/exps_llama_31_8b/meanpooling.py --max-num-workers 8

########################################################
# Qwen 2.5 7B 1M
########################################################
# PBS-Attn
# opencompass ruler/exps_qwen25_7b_1m/pbs.py --max-num-workers 4
# Sparse-Reuse (our method) -- single GPU (no TP support)
# opencompass ruler/exps_qwen25_7b_1m/sparse_reuse.py --max-num-workers 1
# Flash Attention
# opencompass ruler/exps_qwen25_7b_1m/flashattn.py --max-num-workers 4
# Minference
# opencompass ruler/exps_qwen25_7b_1m/minference.py --max-num-workers 4
# FlexPrefill
# opencompass ruler/exps_qwen25_7b_1m/flexprefill.py --max-num-workers 4
# XAttention
# opencompass ruler/exps_qwen25_7b_1m/xattn.py --max-num-workers 4
# MeanPooling
# opencompass ruler/exps_qwen25_7b_1m/meanpool.py --max-num-workers 4
