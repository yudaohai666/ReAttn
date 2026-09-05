from mmengine.config import read_base

with read_base():
    # Datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_128k import \
        needlebench_datasets as needlebench_v2_128k_datasets
    # Summarizer (built-in NeedleBench-V2 summarizer config)
    from opencompass.configs.summarizers.needlebench import \
        needlebench_v2_128k_summarizer
    # Models
    from ...models.qwen3_8b import \
        qwen3_8b_xattn_models as qwen3_8b_xattn_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = 128 * 1024
    model['run_cfg']['num_gpus'] = 1

summarizer = needlebench_v2_128k_summarizer

work_dir = './results/needlebench_v2/exps_qwen3_8b/xattn'
