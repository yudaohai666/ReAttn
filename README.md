# ReAttn / reuse_v1

跨层 block-sparse "reuse" 注意力：为每个 `(layer, kv-head)` 训练一个 Hard Concrete
anchor/sparse 门控，导出为 label（`label.pt`），推理时按 label 走稀疏/复用路径。

本文件记录**经实测验证的推荐训练/推理配置**（Qwen3-8B，RULER）。

---

## 推荐配置（基于 RULER 实测）

以 Qwen3-8B 为例，实测最优组合：

**训练 label**：`tp=0.9`、`target_sparsity=0.8`、`reg_weight=0.003`、`mb8/xb64`
```
attn_patterns/reuse_v1/Qwen3-8B/
  hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt
```

**推理**：`top_p=0.9`、`min_blocks=8`、`max_blocks(xb)=1024`、`last_q_full` 可选（0 或 1）

> 要点：
> - 推理 `top_p` 必须与训练 `top_p` 一致（这里都是 0.9）。
> - 推理 `max_blocks`（xb=1024）可比训练时的 xb64 更宽，放开选块上限。
> - `last_q_full` **不是必需**：开（=1）只在长上下文（128k）有小幅收益（+0.97），
>   64k 几乎无差；关（=0）同样可用。默认可关，追求极限召回再开。

---

## RULER 实测结果

label = Qwen3 `tp=0.9 mb8/xb64`；推理 `top_p=0.9`、`xb=1024`、`min_blocks=8`。
`lqf` = `last_q_full`。每子任务 200 样本，8-GPU。

| subtask | 64k lqf=1 | 64k lqf=0 | 128k lqf=1 | 128k lqf=0 |
|---|---|---|---|---|
| **ruler overall** | **74.69** | **74.42** | **67.34** | **66.37** |
| cwe | 24.35 | 23.80 | 2.20 | 3.10 |
| fwe | 79.17 | 79.17 | 79.50 | 78.50 |
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 | 96.00 | 95.50 |
| niah_single_3 | 100.00 | 100.00 | 98.50 | 96.50 |
| niah_multikey_1 | 87.00 | 86.50 | 81.50 | 80.50 |
| niah_multikey_2 | 58.50 | 56.00 | 58.00 | 53.00 |
| niah_multikey_3 | 21.00 | 17.50 | 5.50 | 6.50 |
| niah_multivalue | 95.25 | 95.00 | 94.38 | 94.50 |
| niah_multiquery | 96.62 | 97.00 | 93.50 | 92.75 |
| qa_squad | 66.50 | 67.50 | 55.00 | 56.00 |
| qa_hotpotqa | 48.00 | 46.50 | 43.00 | 42.50 |
| vt | 94.60 | 98.50 | 68.30 | 63.40 |

结论：`last_q_full` 非必需——开（=1）在 128k 上 +0.97（主要来自 vt / multikey_2 /
single_3），在 64k 上仅 +0.27（且 vt 反而略降）；关（=0）已足够好，长上下文追求
极限召回时再开。

---

## 训练（复现推荐 label）

入口：`scripts/train_reuse_hc.sh`（Hard Concrete，原始 Louizos 2018，固定 L0 惩罚）。

```
bash scripts/train_reuse_hc.sh <model_path> <ctx_len_min> <ctx_len_max> <lr> <num_passkey> \
     [sp_size] [reg_weight] [initial_value] [target_sparsity] [top_p] [min_blocks] [max_blocks]
```

复现上面推荐 label 的取值：

| 参数 | 推荐值 | 脚本默认 | 含义 |
|---|---|---|---|
| `lr` | 0.01 | — | 学习率 |
| `num_passkey` | 10 | — | passkey 数 |
| `ctx_len_min/max` | 8000 / 128000 | — | 训练上下文范围 |
| `sp_size` | 8 | 8 | Ulysses SP group size |
| `reg_weight` | 0.003 | 0.1 | L0 惩罚系数 |
| `initial_value` | 0.0 | 0.0 | 初始 `log_alpha` |
| `target_sparsity` | 0.8 | 0.8 | 导出 top-k cutoff 稀疏比例（仅导出用）|
| `top_p` | 0.9 | 0.7 | topp nucleus 覆盖；**必须与推理 top_p 一致** |
| `min_blocks` | 8 | 8 | topp 选块下限 |
| `max_blocks` | 64 | 64 | topp 上限 / max_sel cache 宽度 |

`torchrun` 固定超参（脚本内写死）：nnodes/nproc=1/8、batch_size=1、grad_accum=1、
max_length=`ctx_len_max`、num_steps=2000、save_steps=50、reg_mode=hc、select_mode=topp、
dataset=`datasets/PaulGrahamEssays.jsonl`(`multiple_passkey`)、needle depth 0.05–0.95、
context_lengths_num_intervals=50、depth_ratio_num_intervals=1000、`--two_pass --no_ac`。

约定：

- Layer 0 强制全 anchor（`log_alpha` 冻结 +10），不进 L0 惩罚与导出 top-k；
  推理拒绝 layer 0 非全 anchor 的 label。
- 输出目录 `attn_patterns/reuse_v1/<model>/<setting>`，
  `setting = hc-orig-rw=<rw>-init=<init>-sp=<sparsity>-tp=<top_p>-lr=<lr>-ctx=<min>_<max>-multi_passkey<N>-sp<sp_size>`；
  仅当 `min_blocks/max_blocks` ≠ 8/64 才追加 `-mb<..>-xb<..>`（老 label 路径不变），`EXP_TAG=...` 可强制追加。
- `RESUME=1` 断点续训（默认关闭）；`WANDB_MODE` 默认 `offline`。

---

## 推理（模型注册与覆盖）

reuse_v1 注册在 `eval/benchmarks/models/{qwen3_8b,llama_31_8b}.py` 的 `*_reuse_v1_models`。
核心 `patch_kwargs`（代码内默认，推理时用环境变量覆盖到上面的推荐值）：

`budget=32`、`block_size=128`、`segment_size=2048`、`sink_blocks=1`、`local_blocks=2`、
`causal=True`、`select_mode=topp`、`per_head_topp=False`；
`torch_dtype=bfloat16`、`max_out_len=2048`、`batch_size=1`、`num_gpus=1`。

环境变量覆盖：

| Qwen3 (`qwen3_8b.py`) | 代码默认 | 推荐(实测) |
|---|---|---|
| `QWEN3_REUSE_V1_LABEL_PATH` | tp=0.7 label | tp=0.9 mb8/xb64 label |
| `QWEN3_REUSE_V1_TOP_P` | 0.7 | 0.9 |
| `QWEN3_REUSE_V1_MAX_BLOCKS` | 64 | 1024 |
| `QWEN3_REUSE_V1_MIN_BLOCKS` | 8 | 8 |
| `QWEN3_REUSE_V1_LAST_Q_FULL` | 1 | 0 或 1（可选，1 有小幅收益）|

Llama-3.1（`llama_31_8b.py`）对应变量：`LLAMA_REUSE_V1_LABEL_PATH`、
`LLAMA_REUSE_V1_TOP_P`(默认 0.9)、`LLAMA_REUSE_V1_LAST_Q_FULL`(默认 1)。

---

## RULER 复现（Qwen3, 8-GPU）

```
RULER_MAX_SEQ_LEN_K=128 QWEN3_REUSE_V1_LAST_Q_FULL=1 \
  bash scripts/eval_ruler_qwen3_tp09_64k.sh
```

脚本已内置推荐推理配置（top_p=0.9、xb=1024、min_blocks=8、tp=0.9 label），可用
`RULER_MAX_SEQ_LEN_K`(64/128) 与 `QWEN3_REUSE_V1_LAST_Q_FULL`(0/1) 覆盖。
离线节点需设 `TIKTOKEN_CACHE_DIR=<repo>/.cache/tiktoken`（cwe/fwe/vt 用 cl100k_base）、
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；opencompass 从 `.venv/bin` 在
`eval/benchmarks/` 下运行，`--max-num-workers 8`。
