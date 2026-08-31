from mmengine.config import read_base

with read_base():
    # Datasets
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_128k import \
        needlebench_datasets as needlebench_v2_128k_datasets
    # Summarizer (built-in NeedleBench-V2 summarizer config)
    from opencompass.configs.summarizers.needlebench import \
        needlebench_v2_128k_summarizer
    # Models
    from ...models.llama_31_8b import \
        llama_31_8b_flexprefill_models as llama3_1_8b_flexprefill_models

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = 128 * 1024
    model['run_cfg']['num_gpus'] = 1

summarizer = needlebench_v2_128k_summarizer

work_dir = './results/needlebench_v2/exps_llama_31_8b/flexprefill'
