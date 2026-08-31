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
bash scripts/train_reuse_kuma.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 8000 128000 0.01 10 8 0.25 anchoridx_4 0.7

# --- 128k, Deterministic Differentiable Pruning gate (ddp-only entrypoint):
#     passes ONLY ddp-relevant args.
#     <model> <ctx_min> <ctx_max> <lr> <num_passkey> [sp_size] [target_sparsity].
#     No reg_weight / desired_density (l1 / kuma only). z_loga init N(1.0,1e-2),
#     3-term learnable-lambda Lagrangian (lr 2e-2/4e-1/2e-2, dual ascent),
#     soft-saturation mean anneals 0.5->0.1 sqrt; export = global top-k head
#     mask hitting exactly target_sparsity. Layer 0 all-anchor. ---
# bash scripts/train_reuse_ddp.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 8000 128000 0.005 10 8 0.7695

# --- 128k, Stochastic Gates (stg-only entrypoint):
#     passes ONLY stg-relevant args.
#     <model> <ctx_min> <ctx_max> <lr> <num_passkey> [sp_size] [target_sparsity] [stg_sigma] [lagrange_lr] [lambda_init_value] [no_ac].
#     anchor_heads = mu_code, init N(0,0.01); z = clip(mu+0.5+sigma*eps, 0, 1);
#     density = mean(Phi(mu/sigma)) over layer 1-31 (layer 0 excluded);
#       => P(z>0.5) per head; tracks true anchor count; aligns with mu>0 threshold.
#     target_sparsity=0.7900 -> desired_density=0.210 -> ~59 total anchor heads.
#     sigma=0.3 (vs 0.5 prev): sharper gate, faster/cleaner polarization.
#     export = global top-k on mu_code hitting exactly target_sparsity. Layer 0 all-anchor.
# (prev runs: stg-sp=0.7944/sigma=0.5 Phi((mu+0.5)/sig); stg-sp=0.7900/sigma=0.5 Phi(mu/sig)) ---
# bash scripts/train_reuse_stg.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 4000 128000 0.01 10 8 0.7900 0.3 0.001 2.0



# bash scripts/train_reuse_hc.sh /root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct 8000 128000 0.01 10 8 0.002 0.0 0.79