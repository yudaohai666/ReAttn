from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.ruler.ruler_cwe_gen import cwe_datasets as _cwe
    from opencompass.configs.datasets.ruler.ruler_fwe_gen import fwe_datasets as _fwe
    from opencompass.configs.datasets.ruler.ruler_niah_gen import niah_datasets as _niah
    from opencompass.configs.datasets.ruler.ruler_qa_gen import qa_datasets as _qa
    from opencompass.configs.datasets.ruler.ruler_vt_gen import vt_datasets as _vt
    from opencompass.configs.summarizers.groups.ruler import \
        ruler_summary_groups
    from ...models.qwen_25_7b_1m import \
        qwen2_5_7b_1m_reuse_v1_models as qwen2_5_7b_1m_reuse_v1_models

_environ = __import__('os').environ

NUM_SAMPLES = int(_environ.get('RULER_NUM_SAMPLES', 100))
_MAX_SEQ_LEN_K = int(_environ.get('RULER_MAX_SEQ_LEN_K', 128))
RULER_MAX_SEQ_LEN = 1024 * _MAX_SEQ_LEN_K
TOKENIZER_MODEL = _environ.get('RULER_TOKENIZER', 'gpt-4')
_TASKS = [t.strip() for t in _environ.get('RULER_TASKS', 'cwe,fwe,niah,qa,vt').split(',') if t.strip()]
_SUBTASKS = [s.strip() for s in _environ.get('RULER_SUBTASKS', '').split(',') if s.strip()]
_LAST_Q_FULL = _environ.get('QWEN25_1M_REUSE_V1_LAST_Q_FULL', '0') == '1'
del _environ
_SUFFIX = f'_{_MAX_SEQ_LEN_K}k'
_GROUPS = dict(cwe=_cwe, fwe=_fwe, niah=_niah, qa=_qa, vt=_vt)

ruler_datasets = []
for _name in _TASKS:
    for _d in _GROUPS[_name]:
        if _SUBTASKS and not any(_s in _d['abbr'] for _s in _SUBTASKS):
            continue
        _t = _d.deepcopy()
        _t['abbr'] = _t['abbr'] + _SUFFIX
        _t['num_samples'] = NUM_SAMPLES
        _t['max_seq_length'] = RULER_MAX_SEQ_LEN
        _t['tokenizer_model'] = TOKENIZER_MODEL
        ruler_datasets.append(_t)

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = sum([v for k, v in locals().items() if k.endswith('_models')], [])

for model in models:
    model['max_seq_len'] = RULER_MAX_SEQ_LEN
    model['run_cfg']['num_gpus'] = 1

summarizer = dict(
    dataset_abbrs=[f'ruler{_SUFFIX}'] + [_d['abbr'] for _d in ruler_datasets],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

work_dir = './results/ruler/exps_qwen25_7b_1m/reuse_v1_lastfull' if _LAST_Q_FULL else './results/ruler/exps_qwen25_7b_1m/reuse_v1'
