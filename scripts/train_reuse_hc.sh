#!/usr/bin/env bash
# Train the reuse_v1 per-(layer, kv-head) anchor/sparse gate with Hard Concrete
# (--reg_mode hc), ORIGINAL Louizos 2018 formulation: fixed L0 penalty.
#
# Hard Concrete (original, no Lagrangian):
#   loss = distill_loss + reg_weight * sum_{all layers} P(z_h != 0)
#        = distill_loss + reg_weight * sum sigmoid(log_alpha_h + 1.599)
#
#   Distill loss pulls important heads to alpha >> 0 (z≈1, anchor).
#   L0 penalty pushes all heads toward alpha << 0 (z≈0, sparse).
#   Two-polar convergence: anchor heads alpha>>0, sparse heads alpha<<0.
#
#   Export: top-k on log_alpha (monotone with sigmoid(log_alpha)) to hit the
#   EXACT target_sparsity, eliminating the sigmoid-tail gap of the raw alpha>0
#   threshold at finite polarization. log_alpha ranking == sigmoid(log_alpha)
#   ranking, so top-k on log_alpha is equivalent to top-k on P(z>0.5).
#
#   Sparsity controlled by reg_weight:
#     larger reg_weight → more sparse    (try 0.2 for ~79% sparse)
#     smaller reg_weight → less sparse   (try 0.05 for ~50% sparse)
#   Start with reg_weight=0.1, adjust by 2x based on observed l0_density.
#   Logged metrics (wandb + progress bar), both over layers 1..L-1:
#     l0_density  = mean P(z != 0) = mean sigmoid(log_alpha + 1.5986)
#                   -- the true Hard Concrete L0 density; tune reg_weight on this
#     write_frac  = mean sigmoid(log_alpha) = expected fraction of heads that
#                   fire the anchor-cache write (rule: z_h > 0.5) this step.
#                   Compare against the deploy anchor fraction over the same
#                   layers: (L*H*(1-target_sparsity) - H) / ((L-1)*H).
#                   For L=32,H=8,target_sparsity=0.8 that is 43/248 = 0.173.
#   target_sparsity is used ONLY at export (top-k cutoff), not during training.
#
# Usage:
#   bash scripts/train_reuse_hc.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> [sp_size] [reg_weight] [initial_value] [target_sparsity] [top_p]
#
#   sp_size          Ulysses SP group size. Default 8.
#   reg_weight       L0 penalty coefficient. Default 0.1.
#   initial_value    Initial log_alpha. Default 0.0.
#   target_sparsity  Fraction of heads to export as sparse (top-k cutoff). Default 0.8.
#   top_p            Nucleus coverage for topp block selection. Default 0.7.
#                    MUST match the inference-time top_p, or the exported head
#                    labels will not transfer.
#
#   Layer 0 is forced all-anchor (log_alpha frozen at +10, requires_grad=False),
#   so it always fills every kv-head slot of the anchor cache. It is excluded
#   from the L0 penalty and from the export top-k. There is no streaming
#   fallback: inference rejects any label whose layer 0 is not all-anchor.
set -euo pipefail

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

model_name=${1}
ctx_len_min=${2}
ctx_len_max=${3}
lr=${4}
num_passkey=${5}
sp_size=${6:-8}
reg_weight=${7:-0.1}
initial_value=${8:-0.0}
target_sparsity=${9:-0.8}
top_p=${10:-0.7}

setting="hc-orig-rw=${reg_weight}-init=${initial_value}-sp=${target_sparsity}-tp=${top_p}-lr=${lr}-ctx=${ctx_len_min}_${ctx_len_max}-multi_passkey${num_passkey}-sp${sp_size}"
exp_name="reuse_v1/${model_name}/${setting}"

torchrun --nnodes 1 --nproc_per_node 8 \
    reuse_v1/train_reuse.py \
    --model_name "${model_name}" \
    --batch_size 1 \
    --max_length "${ctx_len_max}" \
    --dataset_name "datasets/PaulGrahamEssays.jsonl" \
    --dataset_format "multiple_passkey" \
    --num_steps 2000 \
    --lr "${lr}" \
    --reg_mode hc \
    --reg_weight "${reg_weight}" \
    --initial_value "${initial_value}" \
    --target_sparsity "${target_sparsity}" \
    --select_mode topp \
    --top_p "${top_p}" \
    --min_blocks 8 \
    --max_blocks 64 \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min "${ctx_len_min}" \
    --context_length_max "${ctx_len_max}" \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkeys "${num_passkey}" \
    --save_steps 50 \
    --two_pass \
    --no_ac \
    --sp_size "${sp_size}" \
    --output_dir "attn_patterns/${exp_name}"
