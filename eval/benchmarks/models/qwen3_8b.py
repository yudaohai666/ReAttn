from opencompass_models import PatchedHuggingFaceCausalLM

_MODEL_PATH = '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B'

# Label path for reuse_v1. Override at runtime via QWEN3_REUSE_V1_LABEL_PATH.
_REUSE_V1_LABEL_PATH = __import__('os').environ.get(
    'QWEN3_REUSE_V1_LABEL_PATH',
    '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn/attn_patterns/reuse_v1'
    '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Qwen3-8B'
    '/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.7-lr=0.01-ctx=8000_128000-multi_passkey10-sp8/label.pt',
)

_REUSE_V1_LAST_Q_FULL = __import__('os').environ.get('QWEN3_REUSE_V1_LAST_Q_FULL', '1') == '1'
_REUSE_V1_TOP_P = float(__import__('os').environ.get('QWEN3_REUSE_V1_TOP_P', '0.7'))

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

qwen3_8b_flashattn_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen3-8b-flashattn',
        path=_MODEL_PATH,
        patch_type='flashattn',
        patch_kwargs=dict(causal=True),
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        meta_template=api_meta_template,
    ),
]

qwen3_8b_reuse_v1_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen3-8b-reuse-v1',
        path=_MODEL_PATH,
        patch_type='reuse_v1',
        patch_kwargs=dict(
            label_path=_REUSE_V1_LABEL_PATH,
            budget=32,
            block_size=128,
            segment_size=2048,
            sink_blocks=1,
            local_blocks=2,
            causal=True,
            select_mode='topp',
            top_p=_REUSE_V1_TOP_P,
            min_blocks=8,
            max_blocks=64,
            last_q_full=_REUSE_V1_LAST_Q_FULL,
        ),
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        meta_template=api_meta_template,
    ),
]

qwen3_8b_xattn_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen3-8b-xattn',
        path=_MODEL_PATH,
        patch_type='xattention',
        patch_kwargs=dict(
            stride=8,
            threshold=0.9,
            block_size=128,
        ),
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        meta_template=api_meta_template,
    ),
]

qwen3_8b_flexprefill_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen3-8b-flexprefill',
        path=_MODEL_PATH,
        patch_type='flexprefill',
        patch_kwargs=dict(gamma=0.95, tau=0.1),
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        meta_template=api_meta_template,
    ),
]

qwen3_8b_meanpool_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen3-8b-meanpool',
        path=_MODEL_PATH,
        patch_type='meanpooling',
        patch_kwargs=dict(
            block_size=128,
            threshold=0.9,
            force_select_first_block=True,
            force_select_current_block=True,
        ),
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        meta_template=api_meta_template,
    ),
]

qwen3_8b_pbs_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='qwen3-8b-pbs',
        path=_MODEL_PATH,
        patch_type='pbs',
        patch_kwargs=dict(
            block_size=128,
            segment_size=256,
            threshold=0.9,
            force_select_first_block=True,
        ),
        model_kwargs=dict(torch_dtype='torch.bfloat16'),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        meta_template=api_meta_template,
    ),
]
