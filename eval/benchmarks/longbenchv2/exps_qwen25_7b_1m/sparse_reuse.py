from mmengine.config import read_base

with read_base():
    # Models
    # Datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets

    from ...models.qwen_25_7b_1m import \
        qwen2_5_7b_1m_sparse_reuse_models as qwen2_5_7b_1m_sparse_reuse_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    # sparse_reuse loads onto a single device (B==1, custom paged cache); it
    # does not support tensor parallelism, so keep num_gpus=1. Truncate the
    # context to fit one GPU.
    model['max_seq_len'] = 400 * 1024  # truncated to avoid OOM
    model['run_cfg']['num_gpus'] = 1

work_dir = './results/longbenchv2/exps_qwen25_7b_1m/sparse_reuse_400k'
