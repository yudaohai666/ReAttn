#!/usr/bin/env bash
# Launch examples for reuse_v1 anchor/sparse gate training (mirrors
# duo-attention/scripts/run_train_yu.sh). Each line trains one config via
# scripts/train_reuse.sh <model> <ctx_min> <ctx_max> <reg_weight> <lr> <num_passkey> [sp_size].
#
# reuse_v1 is two-pass + activation checkpointing + Ulysses sequence parallelism
# on the full 8xH800 world mesh (SP shards one sequence across sp_size ranks,
# attention all-to-alls internally so top-k block selection stays EXACT; FSDP2
# still shards the 8B weights over the whole world). Llama-3.1-8B (32 q / 8 kv
# heads) -> sp_size=8 (dp_size=1); 128k fits at ~12.8GB/rank with NO CPU offload.
# sp_size=1 falls back to pure FSDP2 DP (each rank runs the full sequence; OOMs
# at 128k -- use sp_size=8 there).

# --- short-context sanity run (fast) ---
# bash scripts/train_reuse.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 1000 4096 0.05 0.02 10

# --- 32k / 64k ---
# bash scripts/train_reuse.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 1000 32768 0.05 0.02 10
# bash scripts/train_reuse.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 1000 65536 0.05 0.02 10

# --- 128k (the target) ---
# bash scripts/train_reuse.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 8000 128000 0.05 0.02 10

# --- 128k, HardKuma gate (kuma-only entrypoint): passes ONLY kuma-relevant args
#     <model> <ctx_min> <ctx_max> <lr> <num_passkey> [sp_size] [desired_density].
#     No reg_weight (L1-only, ignored here); lamda_init=2.0 / lagrange_lr=0.001
#     are fixed to the reference train_kuma_multi_passkey.sh inside the script. ---
# bash scripts/train_reuse_kuma.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 8000 128000 0.01 10 8 0.23 anchoridx_3 0.7

# --- 128k, Hard Concrete gate (hc-only entrypoint): passes ONLY hc-relevant args.
#     <model> <ctx_min> <ctx_max> <lr> <num_passkey> [sp_size] [reg_weight]
#     [initial_value] [target_sparsity] [top_p].
#     anchor_heads = log_alpha; fixed reg_weight L0 penalty sum(P(z>0)) over
#     layers 1..L-1; export = global top-k on log_alpha hitting exactly
#     target_sparsity. Layer 0 is forced all-anchor (log_alpha frozen at +10),
#     excluded from both the penalty and the top-k; no streaming fallback.
#     top_p MUST match the inference-time top_p. ---
bash scripts/train_reuse_hc.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 8000 128000 0.01 10 8 0.002 0.0 0.8
