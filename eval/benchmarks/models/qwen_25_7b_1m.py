from opencompass_models import PatchedHuggingFaceCausalLM

# Checkpoint (label) path for the sparse_reuse model. Override at runtime via the
# QWEN_SPARSE_LABEL_PATH env var (e.g. set it in scripts/*.sh); defaults to full_head.pt.
# Use __import__ to avoid a top-level `import os` (keeps this lazy-import safe).
_SPARSE_LABEL_PATH = __import__('os').environ.get(
    'QWEN_SPARSE_LABEL_PATH',
    '/root/paddlejob/inference-public/yudaohai/sparse_attention/pbs-attn/ckp/Qwen2.5-7B-Instruct-1M/full_head.pt',
)

qwen_25_7b_1m_flashattn_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-flashattn',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='flashattn',
        patch_kwargs=dict(
            causal=True
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

qwen2_5_7b_1m_minference_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-minference',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='minference',
        patch_kwargs=dict(
            vertical_size=1000,
            slash_size=6096,
            adaptive_budget=None
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

qwen2_5_7b_1m_flexprefill_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-flexprefill',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='flexprefill',
        patch_kwargs=dict(
            gamma=0.95,
            tau=0.1,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

qwen2_5_7b_1m_xattn_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-xattn',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='xattention',
        patch_kwargs=dict(
            stride=8,
            threshold=0.9,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

qwen2_5_7b_1m_meanpool_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-meanpool',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='meanpooling',
        patch_kwargs=dict(
            block_size=128,
            threshold=0.9,
            force_select_first_block=True,
            force_select_current_block=True
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

qwen2_5_7b_1m_pbs_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-pbs',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='pbs',
        patch_kwargs=dict(
            block_size=128,
            segment_size=256,
            threshold=0.9,
            force_select_first_block=True,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

qwen2_5_7b_1m_sparse_reuse_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen2_5-7b-instruct-1m-sparse-reuse',
        path='Qwen/Qwen2.5-7B-Instruct-1M',
        patch_type='sparse_reuse',
        patch_kwargs=dict(
            # NOTE: label must be shaped (num_layers, num_kv_heads) for THIS model
            # (Qwen2.5-7B-1M = 28 x 4). Generate one before running; the ckp/
            # labels shipped in this repo are for Llama-3.1-8B and Qwen3-8B only.
            label_path=_SPARSE_LABEL_PATH,
            sparse_phase='prefill',
            sink_blocks=1,
            local_blocks=2,
            topk=32,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]