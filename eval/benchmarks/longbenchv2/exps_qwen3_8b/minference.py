from mmengine.config import read_base

with read_base():
    # Models
    # Datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import \
        LongBenchv2_datasets as LongBenchv2_datasets
        
    from ...models.qwen3_8b import \
        qwen3_8b_minference_models as qwen3_8b_minference_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = 128 * 1024
    model['run_cfg']['num_gpus'] = 1


work_dir = './results/longbenchv2/exps_qwen3_8b/minference'