# 将 DuoAttention 适配为 prefill 阶段稀疏方法的可行性分析与实施方案

> 目标：把 DuoAttention 的「retrieval head 走全量 / streaming head 走 sink+local」思想，
> 作为一个 **prefill 稀疏内核** 接入 pbs-attn_h 现有的 `patch_type` 框架，与
> flashattn / pbs / minference / reuse_v1 等并列，可用同一套 eval 流程测质量和效率。

---

## I. 结论（先说好不好实现）

**难度：低~中，推荐做。**

- 框架高度契合：pbs-attn_h 的 prefill-patch 机制天然支持「按 layer_idx / kv-head 分派不同注意力」，
  而 DuoAttention 的核心就是一张 `(num_layers, num_kv_heads)` 的 full/streaming 分类矩阵。
- 内核现成：本仓 `Block-Sparse-Attention/` 里的 `block_streaming_attn_func` 正是 DuoAttention
  原生训练用的同一个 block-sparse streaming 内核；full head 用 `flash_attn`（本仓已依赖）。
- 只需 **新增 ~2 个文件（核心 <150 行）+ 4 处接线 + 拷贝 1 份 label 数据**，无需改动框架本身。

**主要不确定性（需运行时验证，非阻塞）：**
1. `Block-Sparse-Attention` 的 `.so`（cpython-310）与本仓 venv（torch 2.6.0+cu124）ABI 是否兼容；
2. 该内核对 H800 / SM90 的支持。
- 若上述任一失败，兜底方案 B：改用本仓纯 Triton 的 `Reuse_v1`（sink_blocks/local_blocks 参数天然支持 streaming 掩码）。

---

## II. 原理与「和原生的差异」

### DuoAttention 原生（eval 阶段）
- streaming 的省算力**来自 KV cache 物理压缩**：chunked prefill / decode 时 streaming head 只保留
  `sink_size + recent_size` 长度的 KV（`llama.py:273-290`），而非对全量 KV 做掩码。
- **单趟全量 prefill 是稠密的**（`llama.py:225`，`q_len == kv_seq_len` 分支对所有 head 跑全量 flash_attn）。
- block-sparse 内核在原生里**只在训练时用**。

### 适配到 pbs-attn_h（本方案）
- pbs-attn_h 是 **prefill-only patching**：一次全量 prefill，无 KV reuse / 压缩；decode 走 dense original_forward。
- 因此**不能**照搬「KV 压缩」式稀疏，而是要在**单趟 prefill 内**对 streaming head 施加 sink+local 的
  block-sparse 掩码 —— 这恰好就是 DuoAttention 训练时用的那套 block-streaming 内核。
- full head 仍走全量 `flash_attn_func`。二者按 kv-head 拆分后各算各的，再拼回。

> 语义等价性说明：单趟 prefill 内对 streaming head 施加 sink+local 掩码，和原生「压缩 KV 后再算」
> 在**注意力可见范围**上是一致的（都只看 sink+local），差别仅在原生把不可见的 KV 物理丢弃、
> 本方案保留但掩掉。质量应当对齐，效率上本方案在单趟 prefill 就体现稀疏。

---

## III. 依赖与环境

| 组件 | 位置 | 说明 |
|------|------|------|
| 分类矩阵 `full_attention_heads` | duo 项目 `attn_pattern/<model>/full_attention_heads.tsv` | `(num_layers, num_kv_heads)` 浮点，阈值 0.5 二值化 |
| streaming 内核 | 本仓 `Block-Sparse-Attention/block_sparse_attn/` | `block_streaming_attn_func`，block_size 固定 128，varlen 接口 |
| full 内核 | `flash_attn.flash_attn_func` | 本仓 `pbs_attn/baselines/FlashAttn.py` 已在用 |
| venv | 本仓 `.venv`（py3.10, torch 2.6.0+cu124, H800/SM90） | `.so` 有 cpython-310 变体，py 版本匹配 |

**待验证（首次跑时确认，非阻塞）**：`import block_sparse_attn_cuda` 是否成功；`block_streaming_attn_func`
在 H800 上能否正常前向（跑一个小 shape 对拍 SDPA 参考掩码即可）。

---

## IV. 实现步骤

### Step 0：拷贝 label 数据
把 duo 项目 `attn_pattern/Llama-3.1-8B-Instruct*/full_attention_heads.tsv` 拷进本仓（建议
`ckp/duo/Llama-3.1-8B-Instruct/full_attention_heads.tsv`），供 loader 读取。

### Step 1：新增 `pbs_attn/baselines/duo_pattern.py`（label 加载 + 二值化）
- `load_full_attention_heads(path) -> torch.BoolTensor (num_layers, num_kv_heads)`：
  读 tsv（float），`> 0.5` 得到 bool；True=full(retrieval)，False=streaming。
- 可选：按 `sparsity` 动态阈值（DuoAttention 用分位数选出 streaming 比例 = sparsity），
  即取每层/全局 score 的 `sparsity` 分位点当阈值，复现原生 {0, 0.5, 0.75} 曲线。

### Step 2：新增 `pbs_attn/baselines/DuoAttention.py`（核心 prefill 内核）
函数签名（**形参名必须精确**，patch 靠 `co_varnames` 检测转发）：
```python
def duo_attention_prefill(
    query_states,   # (b, H, q_len, d)  —— 已 RoPE
    key_states,     # (b, Hkv, q_len, d) 或 (b, H, ...) 视框架是否 repeat_kv
    value_states,
    *,
    num_key_value_groups,   # 触发框架保留 native GQA 并传 groups（本内核需要按 kv-head 拆）
    layer_idx,              # 触发框架传入层号 → 查 full_attention_heads[layer_idx]
    holder,                 # ReuseV1Holder 式的 per-model 状态（持有 label 张量、配置）
):
    ...
```
内核逻辑：
1. 从 `holder.full_heads[layer_idx]`（bool, 长度 Hkv）拆出 full / streaming 两组 kv-head 索引。
2. **full 组**：对应 q-heads（每个 kv-head 展开 `num_key_value_groups` 个）走 `flash_attn_func(causal=True)`。
3. **streaming 组**：走 `block_streaming_attn_func`，`head_mask_type=[-1]*n_stream_qheads`，
   `streaming_info=[sink_block_num, local_block_num]*n_stream_qheads`，
   其中 `sink_block_num=ceil(sink_size/128)`、`local_block_num=ceil(recent_size/128)`。
4. 按原始 head 顺序把两组输出散回 `(b, H, q_len, d)`，返回。

> 参考实现：duo 项目 `duo_attn/patch/streaming_attn.py` 的
> `generate_streaming_info_blocksparse_flash_attn` 和 `streaming_attn_blocksparse_flash_attn`
> （含 varlen packing：`(b,s,H,d)->(b*s,H,d)` + `cu_seqlens`），可几乎直接复用。

### Step 3：`get_duo_attention_prefill(...)` 工厂（放 `pbs_attn/patch/huggingface.py`）
```python
def get_duo_attention_prefill(attn_load_dir, sink_size=128, recent_size=256,
                              sparsity=0.5, block_size=128, causal=True, device='cuda'):
    from pbs_attn.baselines.DuoAttention import build_duo_holder, duo_attention_prefill
    holder = build_duo_holder(attn_load_dir, sink_size, recent_size, sparsity, block_size, device)
    return partial(duo_attention_prefill, holder=holder)
```
- 仿 `get_reuse_v1_prefill`：per-model holder（非全局），持有 label 张量与配置。

### Step 4：接线 4 处
1. `pbs_attn/patch/huggingface.py`：加 `get_duo_attention_prefill`（上面 Step 3），
   连同 import 一起放在其余 `get_*_prefill` 附近。
2. `eval/benchmarks/opencompass_models.py`：
   - 顶部 import 里加 `get_duo_attention_prefill`；
   - `_get_prefill_function()` 里加 `elif self.patch_type == 'duo': return get_duo_attention_prefill(**self.patch_kwargs)`。
3. `eval/benchmarks/models/llama_31_8b.py`：新增 `llama_31_8b_duo_models`（照抄 reuse_v1 dict 结构）：
   ```python
   llama_31_8b_duo_models = [dict(
       type=PatchedHuggingFaceCausalLM,
       abbr='llama-3_1-8b-instruct-duo',
       path='.../Llama-3.1-8B-Instruct',
       patch_type='duo',
       patch_kwargs=dict(attn_load_dir=_DUO_LABEL_DIR, sink_size=128,
                         recent_size=256, sparsity=0.5, block_size=128),
       model_kwargs=dict(torch_dtype='torch.bfloat16'),
       max_out_len=2048, batch_size=1, run_cfg=dict(num_gpus=1),
       stop_words=['<|end_of_text|>', '<|eot_id|>'], meta_template=api_meta_template)]
   ```
   （`_DUO_LABEL_DIR` 用 env `LLAMA_DUO_LABEL_DIR` 覆盖，默认指向 Step 0 拷贝路径。）
4. `eval/efficiency/eval_efficiency.py`：加 `duo` method（仿 `ReuseV1Args`），
   `build_prefill_fn` 里透传 `attn_load_dir/sink_size/recent_size/sparsity`。

### Step 5：qwen 同理（可选）
`eval/benchmarks/models/qwen_25_7b_1m.py` 加对应 `qwen_..._duo_models`，label 换 qwen 的 tsv。
框架已支持 Qwen2Attention，无需额外改动。

---

## V. 评估方法

- **质量**：RULER / LongBench-v2，用 `ruler/exps_llama_31_8b/duo.py`（照抄 `pbs.py` 换 models import）。
- **效率**：`scripts/eval_efficiency.sh` 加 `--method duo`，测 E2E prefill 延迟（`generate max_new_tokens=1`）。
  跑法参考记忆里 reuse_v1 的调用（需 `PYTHONPATH=<repo根>`；LongBench-v2 需代理首下）。
- 对比基线：flashattn（上界）/ pbs / reuse_v1 / xattn，同一批 seq len。

---

## VI. 验证 / 对拍

1. **内核可用性**：小 shape（如 s=512, sink=128, recent=256）用 `block_streaming_attn_func`
   vs `generate_streaming_mask` + SDPA 参考掩码对拍，max abs err 应在 bf16 量级（~1e-2）。
2. **端到端连贯性**：Llama-3.1-8B + 一段 ~1.5k token prompt，`sparsity=0.5` 生成应连贯
   （仿 reuse_v1 的 Tier B 验证）。
3. **稀疏度自检**：打印 full/streaming head 计数，确认与 label + sparsity 阈值一致。
4. **sparsity=0 退化**：全 full head，输出应与 flashattn 逐字一致（可作快速回归）。

---

## VII. 风险与兜底

| 风险 | 影响 | 兜底 |
|------|------|------|
| `block_sparse_attn_cuda` ABI 与 torch2.6/cu124 不兼容 | import 失败 | 方案 B：用 `Reuse_v1`（纯 Triton）拼 streaming 掩码 |
| H800/SM90 不支持该 .so | 前向报错 | 同上；或用 SDPA + `generate_streaming_mask`（慢但正确，仅作 quality baseline） |
| tsv label 层/头数与模型不匹配 | 索引越界 | loader 里断言 `full_heads.shape == (num_layers, num_kv_heads)` |
| GQA 展开顺序错位 | 结果错乱 | full/streaming 均在 **kv-head 粒度**拆分，再按 `num_key_value_groups` 展开 q-head，散回时用原始索引 |

**方案 B（纯 Triton 兜底）要点**：`Reuse_v1` 的 `reuse_v1_layer_per_hkv` 已有 `sink_blocks/local_blocks`
和 per-kv-head 的 label 机制；把 DuoAttention 的 streaming head 映射成「只选 sink+local 块、不做 topk」
即可（budget 设为覆盖 sink+local 的定值，或直接走其 streaming 分支）。full head 走 anchor(dense) 路径。

---

## VIII. 工作量估计

- 新文件：`duo_pattern.py`（~40 行）、`DuoAttention.py`（核心 <150 行，含 varlen packing）。
- 接线：4 处小改（huggingface.py / opencompass_models.py / llama_31_8b.py / eval_efficiency.py）。
- 数据：拷 1~2 个 tsv。
- 验证：1 个对拍脚本 + 1 次端到端 smoke。
- 主要时间开销在**首次跑通内核 ABI/SM90 验证**（若失败切方案 B）。

---

## 附：关键文件坐标

| 用途 | 文件 |
|------|------|
| prefill patch 框架 | `pbs_attn/patch/huggingface.py`（`patched_attention_forward` / `get_*_prefill`） |
| patch_type 分派 | `eval/benchmarks/opencompass_models.py`（`_get_prefill_function`） |
| 模型 dict | `eval/benchmarks/models/llama_31_8b.py`（照抄 `llama_31_8b_reuse_v1_models`） |
| full 内核参考 | `pbs_attn/baselines/FlashAttn.py` |
| streaming 内核 | `Block-Sparse-Attention/block_sparse_attn/block_sparse_attn_interface.py`（`block_streaming_attn_func` @行520） |
| duo streaming 参考实现 | duo 项目 `duo_attn/patch/streaming_attn.py`（可直接复用 packing/streaming_info 构造） |
| duo 原生 forward（理解语义） | duo 项目 `duo_attn/patch/llama.py`（`:225` 稠密首趟, `:273-290` KV 压缩） |
| 效率脚本 | `eval/efficiency/eval_efficiency.py` / `scripts/eval_efficiency.sh` |
| label 数据 | duo 项目 `attn_pattern/<model>/full_attention_heads.tsv` |
