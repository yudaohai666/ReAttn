from opencompass_models import PatchedHuggingFaceCausalLM

# Checkpoint (label) path for the sparse_reuse model. Override at runtime via the
# LLAMA_SPARSE_LABEL_PATH env var (e.g. set it in scripts/*.sh); defaults to full_head.pt.
# Use __import__ to avoid a top-level `import os` (keeps this lazy-import safe).
_SPARSE_LABEL_PATH = __import__('os').environ.get(
    'LLAMA_SPARSE_LABEL_PATH',
    '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn/ckp/Llama-3.1-8B-Instruct/full_head.pt',
)

# Prefill dense-tail size for sparse_reuse (last N sparse-head query rows see ALL
# kv). Override at runtime via LLAMA_SPARSE_DENSE_TAIL (e.g. in scripts/*.sh);
# defaults to 32. 0 disables.
_SPARSE_DENSE_TAIL = int(__import__('os').environ.get('LLAMA_SPARSE_DENSE_TAIL', '32'))

# Label path for the reuse_v1 (pure-Triton, H800-capable) model. Points at THIS
# repo's ckp by default (not the pbs-attn one). Override via LLAMA_REUSE_V1_LABEL_PATH.
_REUSE_V1_LABEL_PATH = __import__('os').environ.get(
    'LLAMA_REUSE_V1_LABEL_PATH',
    '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn/ckp/Llama-3.1-8B-Instruct/full_head_new_02.pt',
)

# If '1', the last query block of sparse kv-heads attends to the full KV cache
# (dense attention), improving retrieval recall (e.g. NIAH). Override via
# LLAMA_REUSE_V1_LAST_Q_FULL=0 to disable.
_REUSE_V1_LAST_Q_FULL = __import__('os').environ.get('LLAMA_REUSE_V1_LAST_Q_FULL', '1') == '1'

# Label DIR for the DuoAttention model (holds full_attention_heads.tsv + config.json).
# Override via LLAMA_DUO_LABEL_DIR; defaults to THIS repo's ckp/duo copy.
_DUO_LABEL_DIR = __import__('os').environ.get(
    'LLAMA_DUO_LABEL_DIR',
    '/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/pbs-attn_h/ckp/duo/Llama-3.1-8B-Instruct',
)

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

llama_31_8b_flashattn_models = [
        dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-flashattn',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_minference_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-minference',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_flexprefill_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-flexprefill',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_xattn_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-xattn',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]
llama_31_8b_meanpooling_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-meanpool',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    )
]

llama_31_8b_pbs_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-pbs',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]

llama_31_8b_sparse_reuse_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-sparse-reuse',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
        patch_type='sparse_reuse',
        patch_kwargs=dict(
            label_path=_SPARSE_LABEL_PATH,
            sparse_phase='prefill',
            sink_blocks=1,
            local_blocks=2,
            topk=32,
            dense_tail=_SPARSE_DENSE_TAIL,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]

llama_31_8b_reuse_v1_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-reuse-v1',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
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
            top_p=0.9,
            min_blocks=8,
            max_blocks=64,
            last_q_full=_REUSE_V1_LAST_Q_FULL,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]

llama_31_8b_duo_models = [
    dict(
        type=PatchedHuggingFaceCausalLM,
        abbr='llama-3_1-8b-instruct-duo',
        path='/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct',
        patch_type='duo',
        patch_kwargs=dict(
            attn_load_dir=_DUO_LABEL_DIR,
            sink_size=128,
            recent_size=256,
            sparsity=0.5,
            block_size=128,
            causal=True,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16'
        ),
        max_out_len=2048,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        meta_template=api_meta_template,
    ),
]