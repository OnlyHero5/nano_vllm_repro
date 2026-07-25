# 现状盘点报告 v2 — 2026-07-25

> **与 v1 的关系**：本报告独立于同日早些时候生成的 `current-status-20260725.md`（下称 v1），
> 由新一轮独立诊断产出。v1 的结论被当作"待核实线索"逐条重新验证，结论见 §5。
> 本报告不覆盖、不删除任何历史文档。

> 纯诊断报告：只重建事实基线，不做任何修复、不改任何代码。
> 所有判断附文件路径 + 行号或测试输出作为证据。

---

## 1. 总览结论

### 1.1 整体完成度

| 维度 | 状态 |
|------|------|
| **单卡推理主干**（数据结构 / 模型层 / PagedAttention / Qwen3 / 调度 / 引擎 / 采样） | **代码完整，实测可用**。27 个 Python 文件、3388 行，无 `pass`/`NotImplementedError` 占位 |
| **测试套件** | **20/20 全部通过**（pytest 9.1.1，nano_vllm conda 环境，RTX 4070 Ti） |
| **端到端生成** | **`example.py` 实测通过**：两条 prompt 均生成连贯文本，Decode ≈ 50-120 tok/s |
| **plans_archive 16 篇教程对应功能** | 主干概念已落地（00-04 基础）；**01-04 的增强改动、05-13 全部进阶特性未落地** |
| **代码最后实质修改** | **2026-01-21**（git commit `b0390c4`）。此后全部提交均为文档 |

**一句话**：单卡推理主干代码完整且经实测验证可用；测试全绿（上一轮修复了测试文件的接口 bug）；进阶特性（TP / CUDA Graph / Chunked Prefill / Radix Cache / Speculative Decoding / MoE / FP8 / Offload）零实现；教学文档已三代分叉（plans_archive → experiments → REVIEW_REPORT）。

### 1.2 最值得关注的 3 个发现

1. **v1 报告的核心结论经独立验证全部成立**，仅有一处数值错误（"21个用例 / test_Day4: 4/4"，实测为 **20 个 / Day4 = 3 个**）。v1 对代码状态、测试归因、历史文档校验的判断均可采信。

2. **`block_manager.py:298` 存在真实正确性 bug（off-by-one）**：`len(block_table) > 2` 应为 `>= 2`。当 decode 恰好填满第 2 个块时，链式哈希的 `prefix_hash` 错用 `-1` 而非第 0 块的哈希，导致该块注册了错误的哈希值。后续序列若恰好有相同内容但不同前缀，可能产生**假命中**（复用了 KV 内容不匹配的块）。此 bug 由 `experiments/Day3` 教程首先诊断（v1 独立确认），当前代码仍未修复。

3. **`SamplingParams` 与 `Sampler` 存在 API 矛盾**：`SamplingParams.__post_init__`（sampling_params.py:21）断言 `temperature > 1e-10`，拒绝 `temperature=0`；但 `Sampler.forward`（sampler.py:49-72）专门实现了 greedy 分支（`greedy_mask = temperatures == 0`），且 `test_Day4::test_sampler` 直接测试了 `temps=[0.0, ...]` 并通过。**greedy 路径在 Sampler 层可用，但通过公开 API（SamplingParams → Sequence → ModelRunner）不可达**。

---

## 2. 教程-代码对照矩阵（plans_archive 16 篇）

判定口径：
- ✅ 已完整实现且有测试通过
- 🟡 部分实现，或实现了但无测试覆盖 / 增强未落地
- ⬜ 完全未开始

| 篇 | 主题 | 对应模块 | 状态 | 证据 |
|----|------|---------|------|------|
| **00** | 总览与学习顺序 | （纯导航） | 🟡 | 文档本身无需代码。其技术概念表列出的6 项技术中，PagedAttention / Continuous Batching / Prefix Cache / FlashAttention 已实现；CUDA Graph / Tensor Parallel 未实现。 |
| **01** | WeightLoader 与 Linear 加载协议 | `layers/linear.py`, `utils/loader.py` | 🟡 | **基础协议已实现**：`QKVLinear._weight_loader`（linear.py:76）、`MergedLinear._weight_loader`（:172）、`RowLinear._weight_loader`（:229）、`default_weight_loader`（:238）、`packed_modules_mapping`（qwen3.py:344）、`load_model` 分发（loader.py:33）。**本篇增强未落地**：无 `copy_weight_to_param` 辅助函数（dtype/device 对齐）、`QKVLinear.bias` 未绑 loader（linear.py:69 只绑 weight）。 |
| **02** | Qwen3 主干与权重映射 | `config.py`, `models/qwen3.py` | 🟡 | **主干已实现**：`Qwen3Attention`（qwen3.py:29，GQA + Q/K Norm + RoPE）、`Qwen3MLP`（:175）、`Qwen3DecoderLayer`（:218）、`Qwen3Model`（:288）、`Qwen3ForCausalLM`（:338）。**本篇增强未落地**：`forward()` 仍直接返回 logits（:369-377），无 `compute_logits()`；`Config` 无 `hidden_size` 等 property（config.py 只有 `model` 别名，:62）；注释掉的旧 SDPA 代码仍在（:132-163）。 |
| **02A** | PagedAttention / BlockManager /调度主线（纯概念） | `engine/sequence.py`, `engine/block_manager.py`, `engine/scheduler.py`, `utils/context.py` | ✅ | 纯概念篇。其描述的五个数据结构与生命周期全部已实现并有测试覆盖：`Sequence`（test_Day1::test_sequence ✅）、`BlockManager`（test_Day3 五个用例 ✅）、`Scheduler.schedule()`（无独立测试但 e2e 验证 ✅）、`Context`（test_Day1::test_context ✅）。 |
| **03** | 补全 Sampler 与 SamplingParams | `sampling_params.py`, `layers/sampler.py` | 🟡 | **基础采样已实现**：温度缩放 + Gumbel-Max + greedy（sampler.py:28-74，test_Day4::test_sampler ✅）。**本篇核心增强未落地**：无 `top_k`/`top_p` 字段（sampling_params.py 仅 3 个字段，21 行）；`temperature > 1e-10` 仍拒绝 greedy（:21，本篇要求 `>= 0`）；`Sampler.forward` 不收 top_ks/top_ps。**docstring 与实现不符**：sampler.py:7-9 声称支持 Top-K/Top-P，代码无。 |
| **04** | 串起单卡推理主循环 + Day5 测试 | `engine/model_runner.py`, `engine/llm_engine.py`, `example.py` | 🟡 | **主循环已实现且 e2e 实测通过**：`prepare_prefill/prepare_decode/run`（model_runner.py:173/247/319）、`step/generate`（llm_engine.py:97/129）、`example.py` 跑通。**本篇增强未落地**：`run()` 未拆出 `run_model()`/`prepare_sampling_tensors()`；`run()` 末尾无 `reset_context()`；**`tests/test_Day5.py` 不存在**。 |
| **05** | Tensor Parallel 基础版 | `layers/linear.py`, `models/qwen3.py`, `engine/model_runner.py` | ⬜ | 未开始。无 `rank`/`world_size`/`all_reduce`/Column-RowParallel（grep 零命中）。仅 config.py:25 有 `tensor_parallel_size` 字段 + :50 范围断言；linear.py:200 有注释"未来扩展 TP 时这里需要 all_reduce"。`tests/test_Day6_tp.py` 不存在。 |
| **06** | CUDA Graph 基础版 | `engine/model_runner.py`, `utils/context.py` | ⬜ | 未开始。grep `CUDAGraph`/`capture` 零命中。前置条件 `reset_context()` 已存在（context.py:78，commit b0390c4 "适配计算图录制后的清除操作"），但 graph 本体未写。`config.enforce_eager`（:28）无消费者。`tests/test_Day6_cudagraph.py` 不存在。 |
| **07** | Benchmark 与 Day7 验收 | `bench.py`, `tests/test_Day7.py` | ⬜ | 未开始。`bench.py` 不存在；`tests/test_Day7.py` 不存在；`todo_list.md` 进度表 7 天全标 ⬜（与正文 133 处 ✅ 矛盾）。 |
| **08** | Chunked Prefill 与 v1 调度 | `engine/sequence.py`, `engine/scheduler.py` | ⬜ | 未开始。`Sequence` 无 `num_scheduled_tokens`/`is_prefill` 属性；`Scheduler.schedule()` 仍是整段 prompt 一次性 prefill（scheduler.py:94 `new_tokens = len(seq) - seq.num_cached_tokens`，正是本篇要改的代码）。 |
| **09** | Radix Prefix Cache 与可观测指标 | `engine/block_manager.py` | ⬜ | 未开始。当前是 flat hash 表 prefix cache（block_manager.py:92 `hash_to_block_id`），无 radix/prefix-tree、无命中/复用 token 数等可观测指标。注：基础 prefix cache（块复用 + 引用计数）已实现并通过 test_Day3::test_prefix_cache ✅，但那是 02A 的范围，不是本篇的 Radix 升级。 |
| **10** | Speculative Decoding 基础版 | 新增 `engine/speculative.py` | ⬜ | 未开始。无 draft model、无 verify/accept/reject（grep `speculat`/`draft` 零命中）。`tests/test_Day10_speculative.py` 不存在。 |
| **11** | MoE 推理主线 | `config.py`, `models/qwen3.py`, `utils/loader.py` | ⬜ | 未开始。无 `MoERouter`/`MoEExpert`/`Qwen3MoEMLP`（grep `MoE`/`expert` 零命中）；`Qwen3DecoderLayer` 固定用 dense `Qwen3MLP`（qwen3.py:241）。`tests/test_Day11_moe.py` 不存在。 |
| **11A** | MoE 单卡 Expert-Offloading | `experiments/moe_offloading/` | ⬜ | 未开始。`experiments/` 下只有 Day0-Day7 的 `.md` 教程文件，无 `moe_offloading/` 目录、无 `ExpertWeightCache`。`tests/test_Day11A_offloading.py` 不存在。 |
| **12** | FP8 与 KV Cache 量化 | `utils/kvcache_quant.py`, `engine/model_runner.py` | ⬜ | 未开始。`utils/kvcache_quant.py` 不存在；`Config` 无 `kv_cache_dtype`/`kv_cache_quant_scheme` 字段；`allocate_kv_cache` 只分配裸 `torch.float16`（model_runner.py:126-134）。`tests/test_Day12_kvcache_quant.py` 不存在。 |
| **13** | CPU KV Block Offload | `engine/sequence.py`, `engine/block_manager.py`, `engine/scheduler.py` | ⬜ | 未开始。`SequenceStatus` 只有 WAITING/RUNNING/FINISHED（sequence.py:44-52），无 `SWAPPED`；`BlockManager` 无 `BlockResidency`/`swap_out`/`swap_in`；`Scheduler` 无 `swapped` 队列。`tests/test_Day13_kv_offload.py` 不存在。 |

**矩阵小结**：16 篇中 ✅ 1 篇（02A 纯概念）、🟡 5 篇（00/01/02/03/04，均为"基础已落地、本篇增强未落地"）、⬜ 10 篇（05-13 全部进阶特性 + 11A）。

---

## 3. 测试真实结果（Phase 2 实测）

### 3.1 运行环境

| 项 | 实测值 |
|----|--------|
| Python | 3.12.12（`/home/psx/miniconda3/envs/nano_vllm/bin/python`） |
| GPU | NVIDIA GeForce RTX 4070 Ti，12 GB（`torch.cuda.is_available() == True`） |
| torch | 2.8.0+cu128 |
| flash_attn | 2.8.3 ✅（环境自带） |
| triton | 3.4.0 |
| transformers | 4.57.3 |
| pytest | 9.1.1（本次诊断新装入 nano_vllm 环境） |
| 模型权重 | `models/Qwen3-0.6B/`（1.5 GB）存在 |

### 3.2 pytest 实测结果

```
$ python -m pytest tests -v --tb=short
tests/test_Day1.py::test_sampling_params PASSED
tests/test_Day1.py::test_sequence PASSED
tests/test_Day1.py::test_context PASSED
tests/test_Day1.py::test_config PASSED
tests/test_Day2.py::test_rmsnorm PASSED
tests/test_Day2.py::test_silu_and_mul PASSED
tests/test_Day2.py::test_rope PASSED
tests/test_Day2.py::test_rope_relative_position PASSED
tests/test_Day2.py::test_qwen3_model PASSED
tests/test_Day2.py::test_gqa PASSED
tests/test_Day3.py::test_block PASSED
tests/test_Day3.py::test_block_manager_basic PASSED
tests/test_Day3.py::test_block_manager_append PASSED
tests/test_Day3.py::test_slot_mapping PASSED
tests/test_Day3.py::test_prefix_cache PASSED
tests/test_Day3.py::test_attention_with_context PASSED
tests/test_Day3.py::test_store_kvcache PASSED
tests/test_Day4.py::test_linear_layers PASSED
tests/test_Day4.py::test_sampler PASSED
tests/test_Day4.py::test_sequence_attributes PASSED

============================== 20 passed in 8.69s ==============================
```

**20 个用例全部通过，零失败、零跳过、零报错。**

### 3.3 端到端实测

```
$ python example.py
```

两条 prompt 均成功生成连贯文本：
- "你好，请用300字介绍一下你自己" → 生成带 `\<think\>` 推理格式的完整自我介绍（~300字）
- "1+1=?" → 正确输出 "1 + 1 equals 2."

吞吐：Prefill ≈ 25 tok/s，Decode ≈ 50-120 tok/s（单请求）/ 50-60 tok/s（双请求并发）。进程正常退出。

**结论**：主干 prefill+decode 全链路、权重融合映射、PagedAttention 分页读写、Sampler、调度循环全部正确。

### 3.4 测试覆盖缺口

以下核心模块**无任何测试覆盖**（仅通过 e2e 间接验证）：

| 模块 | 文件 | 说明 |
|------|------|------|
| Scheduler | engine/scheduler.py | 双队列调度、prefill 优先、preemption、postprocess 均无独立测试 |
| ModelRunner | engine/model_runner.py | KV cache 分配、prepare_prefill/decode、run 均无独立测试 |
| LLMEngine | engine/llm_engine.py | 顶层引擎循环无独立测试 |
| Loader | utils/loader.py | 权重加载无独立测试（e2e 间接覆盖） |
| Attention decode 路径 | layers/attention.py | test_Day3 只测了 prefill 路径（`is_prefill=True`），decode 路径（`flash_attn_with_kvcache`）仅 e2e 间接覆盖 |

plans_archive 引用的 `test_Day5.py`、`test_Day6_tp.py`、`test_Day6_cudagraph.py`、`test_Day7.py`、`test_Day9_radix_cache.py`、`test_Day10_speculative.py`、`test_Day11_moe.py`、`test_Day11A_offloading.py`、`test_Day12_kvcache_quant.py`、`test_Day13_kv_offload.py` **全部不存在**。

---

## 4. 代码库现状表（Phase 1）

| 文件 | 行数 | 用途 | 实现状态 | 占位/可疑点 | 测试覆盖 |
|------|------|------|---------|------------|---------|
| `config.py` | 65 | 全局配置 dataclass | ✅完整 | `enforce_eager`（:28）无消费者（无 CUDA Graph）；`tensor_parallel_size`（:25）仅校验范围无实际 TP | test_Day1::test_config ✅ |
| `sampling_params.py` | 21 | 采样参数 | ✅完整（功能窄） | 无 top_k/top_p；`temperature > 1e-10`（:21）拒绝 greedy，与 Sampler 的 greedy 分支矛盾 | test_Day1::test_sampling_params ✅ |
| `llm.py` | 4 | `LLM(LLMEngine)` 一行 wrapper | ✅完整 | — | 无直接测试 |
| `example.py` | 49 | 端到端示例 | ✅完整 | **已实测跑通** | e2e ✅ |
| `engine/sequence.py` | 182 | 请求状态 + block_table | ✅完整 | `block_size=256` 类属性硬编码（:68）；含 `__getstate__/__setstate__` 序列化 | test_Day1::test_sequence ✅、test_Day4::test_sequence_attributes ✅ |
| `engine/block_manager.py` | 329 | PagedAttention 块管理 + hash prefix cache | ✅完整（有 1 个 bug） | **:298 off-by-one**：`len(block_table) > 2` 应为 `>= 2`，破坏恰好 2 块序列的链式哈希（可导致假命中）；flat hash 非 Radix 树 | test_Day3 五个用例 ✅（未触发 :298 路径） |
| `engine/scheduler.py` | 209 | Continuous batching 双队列 + preemption | ✅完整 | preemption 是 recompute 式（:147 deallocate + 放回 waiting 队首），无 swap 式 | **无独立测试**（e2e 间接） |
| `engine/model_runner.py` | 367 | 模型加载 / KV cache / prefill-decode 准备 / run | ✅完整（单卡） | 无 CUDA Graph、无 TP、无 warmup、未拆 run_model；**prepare_prefill 用完整 token_ids（:196），prefix cache 命中不跳过计算（省块不省算力）** | **无独立测试**（e2e 间接） |
| `engine/llm_engine.py` | 191 | 顶层引擎循环 + tqdm 吞吐 | ✅完整 | prefill token 统计含 cached 部分（:123 `sum(len(seq))`），不影响正确性但吞吐数字偏高 | **无独立测试**（e2e 间接） |
| `layers/attention.py` | 220 | Triton store-KV kernel + FlashAttention | ✅完整 | 硬依赖 flash_attn，无 PyTorch fallback；prefill 路径 q/k/v 强制转 fp16（:178-180） | test_Day3::test_attention_with_context ✅（仅 prefill）、test_store_kvcache ✅；**decode 路径无独立测试** |
| `layers/linear.py` | 243 | QKVLinear / MergedLinear / RowLinear | ✅完整（单卡） | **:231 `# TODO: 多卡分片支持`**（唯一代码级 TODO）；无 dtype/device 对齐；QKVLinear.bias 未绑 loader（:69） | test_Day4::test_linear_layers ✅ |
| `layers/layernorm.py` | 113 | RMSNorm + 融合残差 | ✅完整 | docstring 拼写（:82 `redisual`、:85 `normalized_putput`），装饰性 | test_Day2::test_rmsnorm ✅ |
| `layers/activation.py` | 52 | SiluAndMul (SwiGLU) | ✅完整 | — | test_Day2::test_silu_and_mul ✅ |
| `layers/rotary_embedding.py` | 158 | RoPE + get_rope 工厂 | ✅完整 | `assert rope_scaling is None`（:157）明确不支持外推 | test_Day2::test_rope ✅、test_rope_relative_position ✅ |
| `layers/sampler.py` | 95 | 温度 + Gumbel-Max + greedy | ✅完整（与 docstring 不符） | **docstring（:7-9）声称支持 Top-K/Top-P，代码无**；`forward(logits, temperatures)` 不收 top_ks/top_ps | test_Day4::test_sampler ✅ |
| `models/qwen3.py` | 392 | Qwen3ForCausalLM | ✅完整 | **:132-163 大段注释掉的旧 SDPA 实现（死代码）**；`:384 # TODO: 完整映射` 是 from_pretrained 里的遗留注释（实际权重加载在 loader.py，此注释误导）；`:380 mode_path` 参数名笔误 | test_Day2::test_qwen3_model ✅、test_gqa ✅ |
| `utils/context.py` | 80 | 全局 Context 单例 | ✅完整 | `:49 max_context_len: int = None` 类型注解应为 `int \| None`（静态问题，不影响运行） | test_Day1::test_context ✅ |
| `utils/loader.py` | 129 | safetensors → 融合权重加载 | ✅完整 | 无 dtype/device 对齐（直接 `param.data[...].copy_()`） | **无独立测试**（e2e 间接） |

### 高级特性存在性检查（grep 证据）

| 特性 | 状态 | 证据 |
|------|------|------|
| Tensor Parallel | ⬜ 未实现 | 仅 config.py:25 字段 + :50 断言 + linear.py:200 注释。grep `torch.distributed`/`all_reduce`/`world_size` 在代码中零命中 |
| CUDA Graph | ⬜ 未实现 | grep `CUDAGraph`/`capture` 零命中。config.enforce_eager 无消费者 |
| Chunked Prefill | ⬜ 未实现 | grep `chunk` 仅命中 rotary/activation 里的 `torch.chunk`（张量切分，无关） |
| Speculative Decoding | ⬜ 未实现 | grep `speculat`/`draft` 零命中 |
| MoE | ⬜ 未实现 | grep `MoE`/`expert`/`router` 零命中 |
| FP8 / 量化 | ⬜ 未实现 | grep `fp8`/`quant`/`int8` 零命中 |
| GPU Offload / swap | ⬜ 未实现 | grep `offload`/`pin_memory`/`swap` 零命中 |
| Prefix Cache（基础） | 🟡 部分 | block_manager 有链式哈希 + 命中复用 + 引用计数（test_prefix_cache ✅），但是 flat dict 非 Radix 树；prefill 不跳过已缓存 token 计算；无命中统计 |
| Preemption | ✅ 有 | scheduler.__preempt（:147）recompute 式，无 swap 式 |
| 可观测指标 | ⬜ 仅 tqdm | llm_engine.py 的 prefill/decode tok/s，无结构化 metrics |

### 占位标记全仓库扫描

```
$ grep -rn "TODO\|FIXME\|NotImplementedError" --include="*.py" .
models/qwen3.py:141:        # # TODO : flash attention实现    ← 注释块内，死代码
models/qwen3.py:384:        # TODO: 完整映射                  ← 遗留注释，实际加载在 loader.py
layers/linear.py:231:        # TODO: 多卡分片支持              ← RowLinear 未来 TP 扩展
```

**无 `pass` 占位、无 `NotImplementedError`、无 `raise NotImplemented`。** 死代码 1 处（qwen3.py:132-163 注释块）。

---

## 5. 历史分析校验结果（Phase 4）

### 5.0 历史文档时间线

| 时间 | 事件 | 文档/提交 |
|------|------|----------|
| 2026-01-02 | todo_list.md 创建（7 天冲刺计划） | todo_list.md |
| 2026-01-20~21 | **代码密集开发期**（引擎/模型/测试/debug） | commits `62ee47b`→`b0390c4`（最后一次代码提交） |
| 2026-03-09 | Day7 学习教案 + 仓库指南 | commit `286e489`；`.codex-plans/day7-teaching-plans/`（现为空目录） |
| 2026-04-22~23 | plans/ 教学文档重写（repair-plans-docs 任务） | commits `97f5f9c`/`29c7699`/`f0df845`；`.codex-plans/repair-plans-docs/` |
| 2026-04-30 | plans 08-13 + 11-13 重写计划 | commit `bbead9c`；`docs/superpowers/plans/2026-04-30-*.md` |
| 2026-05-01~07 | plans 05-13 文件 mtime（重写执行） | plans_archive/05-13 的 mtime |
| 2026-06-11 | experiments/ 新版教程 + REVIEW_REPORT + plans→plans_archive 重命名 | experiments/*.md、REVIEW_REPORT.md、plans_archive/README.md |
| 2026-07-25 上午 | **v1 盘点**：current-status-20260725.md + tutorial-audit-experiments-20260725.md + 测试文件修复（未提交） | .codex-plans/project-status-analysis/ |
| 2026-07-25 下午 | **本报告（v2）** | 本文件 |

### 5.1 对 v1 报告（current-status-20260725.md）的逐条校验

| v1 结论 | v2 校验结果 |
|---------|------------|
| "20/21 个用例全部通过" | ✅ **基本成立，数值有误**：实测 **20 个**（Day1=4, Day2=6, Day3=7, Day4=**3**），v1 写"21 个 / test_Day4: 4/4"是计数错误 |
| "4 个测试文件有接口 bug，修复后全绿" | ✅ **成立**：git diff 确认 test_Day1-4 有未提交修改（+91/-51），内容为 set_context→Context 对象、Config(model=)→Config(model_path=)、硬编码路径→相对路径、test_Day2 补 Context + CUDA 守卫、test_Day3 store_kvcache 改 4 参合并布局 |
| "代码核心逻辑正确，失败全在测试接口" | ✅ **成立**：20/20 通过 + e2e 通过 |
| "example.py 端到端实测通过" | ✅ **成立**：本次独立复现 |
| "自 2026-01-21 代码零实质改动" | ✅ **成立**：git diff 确认 config.py/linear.py/qwen3.py 仅行尾换行符差异 |
| "REVIEW_REPORT 评审的是 experiments/ 目标态，非当前代码" | ✅ **成立**：REVIEW_REPORT 称 sampling_params.py 59 行含 top_k/top_p（实际 21 行无）、qwen3.py 434 行 forward 返回 hidden states（实际 392 行返回 logits） |
| "repair-plans-docs 修的是文档不是代码" | ✅ **成立**：三份文件均描述 plans/ 文档重写 |
| "todo_list.md 内部矛盾" | ✅ **成立**：正文 133 处 ✅ vs 进度表 7 天全 ⬜ |
| "layers/__init__.py eager import 绑死 flash_attn" | ✅ **成立**：`from .attention import Attention, store_kvcache`（:4）在包导入时触发 `import flash_attn`。在 nano_vllm 环境（有 flash_attn）下不阻断，但无 flash_attn 环境下纯 torch 模块也无法导入 |
| "block_manager.py:298 off-by-one 是真实 bug" | ✅ **成立**：本次独立确认。`len(block_table) > 2` 应为 `>= 2`，decode 填满第 2 块时 prefix_hash 错用 -1 |
| "05-13 全部 ⬜ 未开始" | ✅ **成立**：本次独立 grep 确认 |
| "plans_archive 引用的 test_Day5-13 全部不存在" | ✅ **成立**：tests/ 下只有 test_Day1-4.py |

**v1 总结论全部成立，仅"21 个用例"为计数错误（应为 20 个）。**

### 5.2 对 tutorial-audit-experiments-20260725.md 的校验

| 审计结论 | v2 校验 |
|---------|---------|
| "experiments/ 指南与当前代码高度匹配" | ✅ 成立（v1 已逐条核对，本次未重复） |
| "23 个真实问题覆盖 21 个（91%）" | ✅ 成立 |
| "缺口 #16：layers/__init__.py 全教程未提" | ✅ 成立 |
| "缺口 #20：test_Day2 test_qwen3_model 无 Context，Day2 漏诊" | ✅ 成立（已在上一轮修复） |
| "Day3 §5的 store_kvcache torch.stack 修复方案不可用" | ✅ 成立（stack 复制张量，kernel 写入副本） |
| "Day3 §3 问题 4（block_manager:298 off-by-one）是真实 bug" | ✅ 成立（本次独立确认） |

### 5.3 对 REVIEW_REPORT.md 的校验

**核心判定成立**：REVIEW_REPORT 评审的是 `experiments/` 指南里的"完整代码块"（修复后目标态），不是当前仓库代码。

依然成立的部分：
- 6 个测试 bug 汇总表（已在上一轮修复）
- 与上游 nano-vllm 的接口差异（set_context 单对象、store_kvcache 合并 kv_cache、Scheduler 接收 block_manager、postprocess 2 参、Sequence.block_size 硬编码 256）

不符的部分（v1 已列，本次确认）：
- sampling_params.py 59 行含 top_k/top_p → 实际 21 行无
- sampler.py 186 行 → 实际 95 行
- qwen3.py 434 行 forward 返回 hidden states → 实际 392 行返回 logits
- "SamplingParams temperature >= 0 允许 greedy" → 实际 `> 1e-10` 拒绝

### 5.4 git 历史佐证

- **代码层面**：最后一次功能性代码提交是 `b0390c4`（2026-01-21 "适配计算图录制后的清除操作"，内容仅为 context.py 添加 `reset_context()` 函数）。此后所有提交（2026-03-09、04-22~30）全部是文档。
- **工作区未提交改动**：
  - `config.py`、`layers/linear.py`、`models/qwen3.py`：仅行尾换行符差异，无功能变化
  - `tests/test_Day1-4.py`：上一轮（v1 作者）修复的接口 bug，+91/-51 行，**未提交**
  - `plans/` 整目录被删（D），已移入未跟踪的 `plans_archive/`
  - 未跟踪：`.codex-plans/project-status-analysis/`、`REVIEW_REPORT.md`、`experiments/`、`plans_archive/`、`docs/superpowers/plans/`
- **推论**：`repair-plans-docs/`（04-23）和 `REVIEW_REPORT.md`（06-11）提出的所有"代码修复建议"（dtype 对齐、compute_logits、top_k/top_p 等）**从未被落地到代码**。

---

## 6. 建议的下一步（仅优先级，不含具体修复方案）

> 本轮只诊断。以下供下一轮任务决策参考。

### 如果目标是"让单卡主干真正稳固"

1. **提交上一轮的测试修复**（最高优先）。4 个测试文件的接口 bug 修复（+91/-51 行）目前悬在工作区未提交，是唯一让测试全绿的变更。丢失它测试就回到 4 失败状态。
2. **修 `block_manager.py:298` off-by-one**。这是已确认的正确性 bug，修复只需改一个字符（`>` → `>=`），但不修就始终有假命中风险。同时补一个触发恰好 2 块场景的测试用例。
3. **解决 `SamplingParams` 与 `Sampler` 的 greedy 矛盾**。要么让 `SamplingParams` 接受 `temperature=0`（改 `> 1e-10` 为 `>= 0`），要么删掉 Sampler 里的 greedy 分支。当前状态是"Sampler 说我能 greedy，API 说不行"。
4. **给 Scheduler / ModelRunner / Attention decode 路径补独立测试**。这三个模块目前只有 e2e 间接覆盖，任何回归都无法被测试套件捕获。

### 如果目标是"继续往进阶特性走"

5. **先补 01-04 的增强改动**（dtype 对齐、compute_logits 拆分、top_k/top_p、run_model 拆分），因为 05(TP)/06(CUDA Graph) 显式依赖它们（教程前置条件写明）。
6. **统一并冻结"哪套文档算数"**。当前 plans_archive（16 篇）、experiments（Day0-7）、REVIEW_REPORT、todo_list 四套叙述互相打架。建议明确：以哪套为唯一事实来源、plans→plans_archive 重命名是否要提交、REVIEW_REPORT 标注为"评审 experiments 指南、非当前代码"。

### 如果目标是"审计教程文档质量"

7. **experiments/ 教程已有高质量审计**（tutorial-audit-experiments-20260725.md），结论可信。plans_archive16 篇的逐篇代码匹配审计尚未做过（本报告 §2 是功能落地矩阵，不是代码片段级匹配），若需要可作为下一轮任务。

---

## 附：本次盘点未覆盖 / 待查项

- `plans_archive/` 内部跨篇相对链接在 `plans/`→`plans_archive/` 重命名后是否仍有效，未逐一验证。
- `experiments/` Day0-7 八篇指南的代码片段与当前代码的逐行匹配（tutorial-audit已做，本次未重复）。
- `layers/__init__.py` eager import 在无 flash_attn 环境下的实际影响（本次在 nano_vllm 环境测试，flash_attn 已就绪，未复现阻断）。
- `.codex-plans/day7-teaching-plans/` 空目录的来历（git 无历史，可能是手工创建后清空）。
