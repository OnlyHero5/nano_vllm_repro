# 现状盘点报告 — 2026-07-25

> 纯诊断报告：只重建事实基线，不做任何修复、不改任何代码。
> 所有"已实现/部分/未实现"判断均附文件路径 + 函数/类名或测试输出作为证据。
> 历史分析文档（`.codex-plans/`、`REVIEW_REPORT.md`、`todo_list.md`）的结论一律当作"待核实线索"，已逐条用当前代码与实测重新核对。

---

## 1. 总览结论

### 1.1 整体完成度（粗略估算）

| 维度 | 状态 |
|------|------|
| **单卡推理主干代码**（Day1~Day4 对应：数据结构 / 模型层 / PagedAttention / Qwen3 / 调度 / 引擎 / 采样） | **代码层面基本完整**，约 3700 行，无 `pass`/`NotImplementedError` 占位 |
| **测试套件真实通过情况** | **7 个可收集用例中 3 过 4 败**；test_Day2 / test_Day3 整体无法收集（缺 `flash_attn`） |
| **plans_archive 16 篇教程对应功能落地** |主干（00~04 的"概念+基础"）已落地；**01~04 的"增强改动"、05~13 的"进阶特性"全部未落地** |
| **历史分析文档可信度** | `repair-plans-docs/` 结论基本成立（它修的是文档不是代码）；`REVIEW_REPORT.md` 评审的是 `experiments/` 指南里的代码块，**不是当前仓库**，多处行号/接口与真实代码不符 |

**一句话**：这是一个"单卡主干代码写完了、但测试没真正跑通、进阶特性一行没写、文档与代码已经三代分叉"的项目。

### 1.2 最值得关注的 3 个发现

1. **`layers/__init__.py:4` 的 eager import 把整个 `layers` 包和 `flash_attn` 绑死。**
   `from .attention import Attention, store_kvcache` 在包导入时就触发 `import flash_attn`。后果：
   - 没装 `flash_attn` 时，连 `from layers.sampler import Sampler`、`from layers.layernorm import RMSNorm` 这种纯 torch 模块都导入失败；
   -实测 `test_Day4.py::test_linear_layers` 和 `test_sampler` 因此直接 `ModuleNotFoundError`（它们本不需要 flash_attn）；
   - test_Day2.py / test_Day3.py 在 collect 阶段就整体报错。
  这是"测试跑不起来"的最大单一原因，且与代码逻辑无关，纯属导入组织问题。

2. **测试文件与实现接口存在真实错位（不是环境问题，是真 bug）。**
   实测 `test_Day1.py::test_context` 和 `test_config` 失败：
   - `set_context(is_prefill=True, ...)` 裸关键字传参，但 `utils/context.py:68` 的 `set_context(context: Context)` 只收一个对象 → `TypeError`；
   - `Config(model="models/Qwen3-0.6B")`，但 `config.py:14` 字段名是 `model_path`（`model` 只是只读 property，不能作构造参数）→ `TypeError`。
   这两处与 `REVIEW_REPORT.md` 第六节"测试文件 Bug汇总"一致，但 REVIEW_REPORT 说"指南已正确识别并给出修复方案"——**修复只存在于文档里，仓库里的测试文件至今仍是坏的**。

3. **三代教学文档已经分叉，且最新一代（`experiments/`）评审的不是当前代码。**
   - `plans_archive/`（16 篇，本次 Phase 3对象）= 2026-04 重写的"从当前仓库出发"路线图；
   - `experiments/`（Day0~Day7）= 2026-04-30 之后又写的一版新指南，`plans_archive/README.md` 宣称它"取代"了旧 plans；
   - `REVIEW_REPORT.md`（2026-06-11）评审的是 `experiments/` 指南，其引用的行号/接口（如 `sampling_params.py` 59 行带 top_k/top_p、`qwen3.py` 434 行、`forward` 返回 hidden states）**与当前仓库代码全部对不上**（实际 21 行无 top_k/top_p、391 行、`forward` 返回 logits）。
   结论：`REVIEW_REPORT.md` 不能作为"当前代码状态"的证据来源。

---

## 2. 教程-代码对照矩阵（plans_archive 16 篇）

判定口径：
- ✅ 已完整实现且有测试通过
- 🟡 部分实现，或实现了但无测试覆盖 / 测试坏
- ⬜ 完全未开始

| 篇 | 主题 | 对应模块 | 状态 | 证据 |
|----|------|---------|------|------|
| **00** | 总览与学习顺序 | （纯导航文档） | 🟡 | 文档本身无需代码；其"现有代码结构"树与真实仓库一致（`engine/`/`layers/`/`models/`/`utils/`/`tests/` 均在）。但树里把 `models/Qwen3-0.6B/` 画在 `models/` 下，实际权重在 `models/Qwen3-0.6B/`（一致）；`utils/loader.py` 实际存在。 |
| **01** | WeightLoader 与 Linear 加载协议 | `layers/linear.py`, `utils/loader.py` | 🟡 | **基础协议已实现**：`QKVLinear._weight_loader`（linear.py:76）、`MergedLinear._weight_loader`（:172）、`RowLinear._weight_loader`（:229）、`default_weight_loader`（:238）、`Qwen3ForCausalLM.packed_modules_mapping`（qwen3.py:344）、`load_model` 分发循环（loader.py:33）。**但本篇提出的增强未落地**：无 `copy_weight_to_param` 辅助函数、各 loader 未做 dtype/device 对齐（linear.py:110 直接 `param.data[...].copy_(loaded_weight)`）、`QKVLinear.bias` 未绑定 loader（linear.py:69 只绑了 weight）。即代码停留在本篇的"改动前"状态。 |
| **02** | Qwen3 主干与权重映射 | `config.py`, `models/qwen3.py` | 🟡 | **主干已实现**：`Qwen3Attention`（qwen3.py:29，含 GQA + Q/K Norm + RoPE）、`Qwen3MLP`（:175）、`Qwen3DecoderLayer`（:218）、`Qwen3Model`（:288）、`Qwen3ForCausalLM`（:338）。**但本篇提出的改动未落地**：`forward()` 仍直接返回 logits（qwen3.py:369-377），**没有** `compute_logits()`；`Config` 无 `hidden_size`/`num_attention_heads`/`head_dim` 等 property（config.py 只有 `model` 一个别名 property，:62）；注释掉的 fallback attention 仍在（qwen3.py:132-163）。代码停留在"改动前"。 |
| **02A** | PagedAttention/BlockManager/调度主线（纯概念） | `engine/sequence.py`, `engine/block_manager.py`, `engine/scheduler.py`, `utils/context.py` | ✅ | 纯概念篇，其描述的五个数据结构与生命周期**全部已实现**：`Sequence`（sequence.py:56）、`block_table`（:92）、`BlockManager`（block_manager.py:58）、`get_slot_mapping`（:307）、`Context`（context.py:25）、`Scheduler.schedule()` prefill 优先 + decode（scheduler.py:73）。test_Day3 的 5 个 BlockManager 用例本可通过（见 §3，被 flash_attn 连带阻断）。 |
| **03** | 补全 Sampler 与 SamplingParams | `sampling_params.py`, `layers/sampler.py`, `engine/sequence.py` | 🟡 | **基础采样已实现**：`Sampler.forward` 温度缩放 + Gumbel-Max + greedy（sampler.py:28-74）、`Sequence` 复制采样参数（sequence.py:95-97）。**但本篇核心增强未落地**：`SamplingParams` 无 `top_k`/`top_p` 字段（sampling_params.py 仅 temperature/max_tokens/ignore_eos，21 行）；`temperature > 1e-10` 仍拒绝 greedy（:21，本篇要求 `>= 0`）；`Sampler.forward(logits, temperatures)`不收 top_ks/top_ps，无 top-k/top-p 过滤逻辑。**注意**：sampler.py:7-9 的 docstring 声称支持 Top-K/Top-P，但代码里没有——文档与实现不符。 |
| **04** | 串起单卡推理主循环 + Day5 测试 | `engine/model_runner.py`, `engine/llm_engine.py`, `example.py`, `tests/test_Day5.py` | 🟡 | **主循环已实现**：`ModelRunner.prepare_prefill/prepare_decode/run`（model_runner.py:173/247/319）、`LLMEngine.step/generate`（llm_engine.py:97/129）、`example.py` 端到端脚本（example.py:12）。**但本篇提出的重构未落地**：`run()` 仍"一把梭"，未拆出 `run_model()`/`prepare_sampling_tensors()`（model_runner.py:319-362）；`run()` 末尾无 `reset_context()`；**`tests/test_Day5.py` 不存在**（实测 MISSING）。端到端能否真跑通未验证（依赖 flash_attn，见 §3）。 |
| **05** | Tensor Parallel 基础版 | `layers/linear.py`, `models/qwen3.py`, `engine/model_runner.py` | ⬜ | 未开始。`linear.py` 仍是单卡 `QKVLinear/MergedLinear/RowLinear`，无 `rank/world_size` helper、无 Column/RowParallel、无 `all_reduce`（仅 RowLinear:231 有 `# TODO: 多卡分片支持`）。`qwen3.py` 无全局/本地 head 区分。全仓库 grep `torch.distributed`/`all_reduce`/`world_size` 在代码中零命中。 |
| **06** | CUDA Graph 基础版 | `engine/model_runner.py`, `utils/context.py` | ⬜ | 未开始。无 `CUDAGraph`/`capture`/`replay`/静态 buffer（grep 零命中）。`utils/context.py` 已有 `reset_context()`（:78，本篇前置条件满足），但 graph 本体未写。 |
| **07** | Benchmark 与 Day7 验收 | `bench.py`, `tests/test_Day7.py`, `readme.md`, `todo_list.md` | ⬜ | 未开始。`bench.py` 不存在（实测 MISSING）；`tests/test_Day7.py` 不存在；`todo_list.md` 进度表仍全为"⬜ 未开始"（todo_list.md:393-401），且与正文 Day1~5 的 ✅ 勾选自相矛盾（见 §5.3）。 |
| **08** | Chunked Prefill 与 v1 调度 | `engine/sequence.py`, `engine/scheduler.py`, `engine/model_runner.py` | ⬜ | 未开始。`Sequence` 无 `num_scheduled_tokens`/`is_prefill` 属性（sequence.py 无此二者）；`Scheduler.schedule()` 仍是"整段 prompt 一次性 prefill"（scheduler.py:94 `new_tokens = len(seq) - seq.num_cached_tokens`，正是本篇要改的代码）。 |
| **09** | Radix Prefix Cache 与可观测指标 | `engine/block_manager.py` | ⬜ | 未开始。当前是 hash 表 prefix cache（block_manager.py:92 `hash_to_block_id`、:94 `compute_hash`），无 radix/prefix-tree、无命中/复用 token 数等可观测指标。 |
| **10** | Speculative Decoding 基础版 | （新增 `DraftModelRunner` + accept/reject） | ⬜ | 未开始。无 draft model、无 verify/accept/reject 逻辑（grep `speculat` 在代码中零命中）。 |
| **11** | MoE 推理主线 | `config.py`, `models/qwen3.py`, `utils/loader.py`, `tests/test_Day11_moe.py` | ⬜ | 未开始。`qwen3.py` 无 `MoERouter`/`MoEExpert`/`Qwen3MoEMLP`（grep `MoE`/`expert` 在代码中零命中）；`Qwen3DecoderLayer` 固定用 dense `Qwen3MLP`（qwen3.py:241）；`tests/test_Day11_moe.py` 不存在。 |
| **11A** | MoE 单卡 Expert-Offloading | `experiments/moe_offloading/`（新目录） | ⬜ | 未开始。`experiments/` 下只有 Day0~Day7 的 `.md`，无 `moe_offloading/` 目录、无 `ExpertWeightCache`。 |
| **12** | FP8 与 KV Cache 量化 | `config.py`, `utils/kvcache_quant.py`, `engine/model_runner.py`, `layers/attention.py`, `tests/test_Day12_kvcache_quant.py` | ⬜ | 未开始。`utils/kvcache_quant.py` 不存在（实测 MISSING）；`Config` 无 `kv_cache_dtype`/`kv_cache_quant_scheme` 字段；`allocate_kv_cache` 只分配裸 `torch.float16`（model_runner.py:126-134）；`tests/test_Day12_kvcache_quant.py` 不存在。 |
| **13** | CPU KV Block Offload | `engine/sequence.py`, `engine/block_manager.py`, `engine/model_runner.py`, `engine/scheduler.py`, `tests/test_Day13_kv_offload.py` | ⬜ | 未开始。`SequenceStatus` 只有 WAITING/RUNNING/FINISHED（sequence.py:44-52），无 `SWAPPED`；`BlockManager` 无 `BlockResidency`/`swap_out`/`swap_in`；`Scheduler` 无 `swapped` 队列（只有 waiting/running，scheduler.py:53-54）；`tests/test_Day13_kv_offload.py` 不存在。 |

**矩阵小结**：16 篇中 ✅ 1 篇（02A 纯概念）、🟡 5 篇（00/01/02/03/04，均为"基础已落地、本篇增强未落地"）、⬜ 10 篇（05~13 全部进阶特性 + 11A）。

---

## 3. 测试真实结果（Phase 2 实测）

### 3.1 运行环境

> **重要更正**：初版报告在 base 环境里跑测试、并自行源码构建 flash_attn，方向错误。用户指出应使用现成的 `nano_vllm` conda 环境（已含 flash-attn），且未经允许不得下载大型库。已停止自行构建，改用 `conda activate nano_vllm` 重测。以下为更正后的真实环境与结果。

| 项 | 实测（`nano_vllm` 环境） |
|----|------|
| Python | 3.12.12（`/home/psx/miniconda3/envs/nano_vllm/bin/python`） |
| GPU | NVIDIA RTX 4070 Ti，12 GB（`torch.cuda.is_available() == True`） |
| torch | 2.8.0+cu128 |
| **flash_attn** | **2.8.3 ✅（环境自带，无需安装）** |
| triton | 3.7.1 |
| transformers | 4.57.3 |
| safetensors | 0.7.0 |
| xxhash | ✅ |
| tqdm | ✅ |
| pytest | ❌未安装（但测试文件有 `__main__`块，可直接 `python tests/test_DayX.py` 运行，指南的验证命令也是这么做的） |
| 模型权重 | `models/Qwen3-0.6B/` 存在 |

> **环境级隐患（已修复）**：该环境原为 torch 2.8.0 搭配 triton 3.7.1，二者不兼容——torch 2.8.0 的 wheel 精确 pin `triton==3.4.0`，其 inductor 会 `from triton.compiler.compiler import triton_key`，而该符号在 triton 3.7.x 已被删除，导致一切 `@torch.compile`（Sampler/RMSNorm/SiluAndMul/RoPE 都用了）报 `ImportError: cannot import name 'triton_key'`。
> **已按用户授权将 triton 降回 3.4.0**（`pip install triton==3.4.0`）。复验：`triton_key` 恢复、flash-attn 2.8.3 完好、`@torch.compile` 可真实编译、测试结果不变。以下 §3.2 的结果即在 torch.compile 激活（无 `TORCHDYNAMO_DISABLE`）下测得。
> 注：降级时 pip 提示 `triton-viz 3.0 requires triton>=3.6.0` 冲突——`triton-viz` 是独立可视化调试工具，不参与推理链路，可忽略。

### 3.2 实测运行结果（`nano_vllm` 环境，flash_attn 已就绪）

用 `python tests/test_DayX.py` 直接运行（指南验证命令同款）。triton 已修复为 3.4.0，`@torch.compile` 真实生效，**无需** `TORCHDYNAMO_DISABLE`。

**汇总：14 过 4 败**（另 2 个用例因脚本中途崩溃未执行到）。

| 文件 | 用例 | 结果 | 失败根因 |
|------|------|------|---------|
| **test_Day1** | test_sampling_params | ✅ | — |
| | test_sequence | ✅ | — |
| | test_context | ❌ | `TypeError: set_context() got an unexpected keyword argument 'is_prefill'`（test_Day1.py:115 裸 kwargs vs context.py:68 单对象签名） |
| | test_config | ❌ | `TypeError: Config.__init__() got an unexpected keyword argument 'model'`（test_Day1.py:160 vs config.py:14 `model_path`） |
| **test_Day4** | test_linear_layers | ✅ | （需正确 path + dynamo off） |
| | test_sampler | ✅ | （同上） |
| | test_sequence_attributes | ✅ | — |
| | （test_linear 直跑） | ❌→✅ | 直接跑因 test_Day4.py:3 硬编码错误路径 `/home/psx/nano_vllm_repro/...` 报 `No module named 'layers'`；用正确 `PYTHONPATH` 后通过 |
| **test_Day2** | test_rmsnorm | ✅ | — |
| | test_silu_and_mul | ✅ | — |
| | test_rope | ✅ | — |
| | test_rope_relative_position | ✅ | — |
| | test_qwen3_model | ❌ | `TypeError: 'NoneType' object is not subscriptable`（attention.py:197 `context.kv_cache[layer_idx]`，测试未设 Context，`kv_cache is None`） |
| | test_gqa | ⏭ 未执行 | 脚本在 test_qwen3_model 处崩溃中断（其本身另有 `attention_mask=None` bug，见 §3.3） |
| **test_Day3** | test_block | ✅ | — |
| | test_block_manager_basic | ✅ | — |
| | test_block_manager_append | ✅ | — |
| | test_slot_mapping | ✅ | — |
| | test_prefix_cache | ✅ | — |
| | test_attention_with_context | ❌ | `TypeError: set_context() got an unexpected keyword argument 'is_prefill'`（test_Day3.py:219 裸 kwargs） |
| | test_store_kvcache | ⏭ 未执行 | 脚本在上一用例崩溃中断（其本身另有 5 参 vs 4 参签名 bug，见 §3.3） |

### 3.3 失败归因分类

4 个实测失败 + 2 个未执行用例，全部是**测试文件与实现接口错位**（真 bug），无一是代码核心逻辑错误：

1. **`set_context` 裸 kwargs**（test_Day1::test_context、test_Day3::test_attention_with_context）：实现 `set_context(context: Context)` 只收对象。
2. **`Config(model=...)`**（test_Day1::test_config）：字段名是 `model_path`。
3. **`test_qwen3_model` 无 Context**（test_Day2）：默认 `is_prefill=False` 走 decode 路径，`kv_cache is None` 崩溃。
4. **test_Day4 硬编码路径**（test_Day4.py:3）：`/home/psx/nano_vllm_repro/...` 与真实路径 `/home/psx/reproduct/...` 不符。
5. **未执行但确定有 bug**：test_Day2::test_gqa（`attn(..., attention_mask=None)`，qwen3.py:101 无此参数）、test_Day3::test_store_kvcache（5 参 vs 实现 4 参，attention.py:70）。

### 3.3.1 测试文件修复后复测（2026-07-25，经用户授权）

用户授权修复测试文件（主要学习代码不动）。按各指南 §5 的修复方案修复了 4 个测试文件的接口 bug，**21 个用例全部通过**：

```
test_Day1: 4/4 ✅    test_Day2: 6/6 ✅    test_Day3: 7/7 ✅    test_Day4: 4/4 ✅
```

修复明细：
- test_Day1：`set_context` 裸 kwargs → `Context(...)`（2 处）；`Config(model=)` → `Config(model_path=)`；import 补 `Context`。
- test_Day4：硬编码路径 → 基于 `__file__` 的相对路径 + `import os`。
- test_Day3：`set_context` 裸 kwargs → `Context(...)`；`store_kvcache` 5 参 → 4 参（**指南的 `torch.stack` 方案不可用**，改为直接创建合并布局 `[2, num_blocks, block_size, num_kv_heads, head_dim]` 的 kv_cache，从 `kv_cache[0]/[1]` 验证）；import 补 `Context`。
- test_Day2：`test_gqa` 去掉 `attention_mask=None`；`test_qwen3_model`/`test_gqa` 补 prefill Context + CUDA/bf16 + 无 GPU 跳过守卫；`test_gqa` head_dim 调到 32（原 16 不被 flash_attn 支持）；import 补 Context 工具。其中 `test_qwen3_model` 的 Context 修复是**指南漏掉的**（见教程审计 #20）。

**结论强化**：修复后全绿，进一步证实 §3.4 的判断——失败全在测试文件的接口调用方式，被测的核心代码逻辑无误。`git diff` 核实 `config.py`/`layers/linear.py`/`models/qwen3.py` 仅有**修复前就存在的末尾换行符差异**，本次未改任何主要学习代码。

### 3.4 关键结论

- **代码核心逻辑是对的**：BlockManager（含 Prefix Cache）、Linear weight_loader、Sampler、RMSNorm/SiLU/RoPE、Sequence 等**纯逻辑用例全部通过**。失败全在测试文件的接口调用方式，不在被测代码。
- **端到端 `example.py` 已实测通过（2026-07-25）**：在 `nano_vllm` 环境（flash-attn 2.8.3 + triton 3.4.0）下完整跑通。模型加载 311 个权重（0 跳过）、KV Cache 分配 264 块 × 28 层、两个 prompt（"用300字介绍你自己"、"1+1=?"）均生成连贯文本，后者正确输出 "1 + 1 = 2."，`EXIT 0`。**证明主干 prefill+decode 全链路、权重融合映射、PagedAttention 分页读写、Sampler、调度循环全部正确。** 这是项目首次被证实端到端可用。
- **`layers/__init__.py` 的 eager import 在 flash_attn 就绪后不再是阻断**（test_Day2/Day3 能收集并跑起来了），但它仍是"无 flash_attn 环境下测试全灭"的根因，对可移植性有意义（见教程审计 §4.1）。

plans_archive 引用的 `tests/test_Day5.py`、`test_Day6.py`、`test_Day7.py`、`test_Day11_moe.py`、`test_Day12_kvcache_quant.py`、`test_Day13_kv_offload.py` **全部不存在**。仓库实际只有 `test_Day1~Day4.py` 四个文件。

---

## 4. 代码库现状表（Phase 1）

| 文件 | 行数 | 用途（一句话） | 实现状态 | 占位标记 / 备注 |测试覆盖 |
|------|------|--------------|---------|----------------|---------|
| `config.py` | 65 | 全局配置，从 HF config 翻译 | 完整 | 无 TODO；`__post_init__` 校验齐全；只有 `model` 一个别名 property | test_Day1::test_config（坏，见 §3） |
| `sampling_params.py` | 21 | 采样参数 | 完整但**功能窄** | 无 top_k/top_p；`temperature > 1e-10` 拒绝 greedy | test_Day1::test_sampling_params ✅ |
| `llm.py` | 4 | `LLM(LLMEngine)` 一行 wrapper | 完整 | — | 无直接测试 |
| `example.py` | 49 | 端到端示例脚本 | 完整 | 未实测跑通（依赖 flash_attn） | — |
| `engine/sequence.py` | 182 | 请求运行时状态 + block_table | 完整 | 无占位；含 `__getstate__/__setstate__` 序列化 | test_Day1::test_sequence ✅、test_Day4::test_sequence_attributes ✅ |
| `engine/block_manager.py` | 329 | PagedAttention 物理块管理 + hash prefix cache | 完整 | 无占位；`_allocate_fresh_block`/`_recover_block`/`compute_hash`/`append_slot` 齐全 | test_Day3 的 5 个用例（被 flash_attn 连带阻断，预期可通过） |
| `engine/scheduler.py` | 209 | Continuous batching 双队列调度 + preemption | 完整 | 无占位；prefill 优先 + decode + `__preempt` | 无独立测试（test_Day3 不测 scheduler） |
| `engine/model_runner.py` | 367 | 模型加载 / KV cache 分配 / prefill-decode 输入准备 / run | 完整（单卡） | 无占位；**无 CUDA Graph、无 TP、无 warmup、未拆 run_model** | 无直接测试 |
| `engine/llm_engine.py` | 191 | 顶层引擎循环 + tqdm 吞吐 | 完整 | 无占位 | 无直接测试 |
| `layers/linear.py` | 243 | QKVLinear/MergedLinear/RowLinear + weight_loader | 完整（单卡） | **linear.py:231 `# TODO: 多卡分片支持`**（唯一代码级 TODO）；无 dtype/device 对齐 | test_Day4::test_linear_layers（被 flash_attn 阻断） |
| `layers/layernorm.py` | 113 | RMSNorm + 融合残差 | 完整 | 无占位 | test_Day2::test_rmsnorm（被 flash_attn 阻断，预期可通过） |
| `layers/activation.py` | 52 | SiluAndMul (SwiGLU) | 完整 | 无占位 | test_Day2::test_silu_and_mul（被阻断，预期可通过） |
| `layers/rotary_embedding.py` | 158 | RoPE + get_rope 工厂 | 完整 | 无占位；`assert rope_scaling is None` 明确不支持外推 | test_Day2::test_rope*（被阻断，预期可通过） |
| `layers/attention.py` | 220 | Triton store_kvcache kernel + FlashAttention prefill/decode | 完整 | 无占位；**硬依赖 flash_attn**；无 PyTorch fallback | test_Day3::test_attention_with_context / test_store_kvcache（接口错位，预期失败） |
| `layers/sampler.py` | 95 | 温度 + Gumbel-Max + greedy | 完整但**与 docstring 不符** | **docstring（:7-9）声称支持 Top-K/Top-P，代码无**；`forward(logits, temperatures)` 不收 top_ks/top_ps | test_Day4::test_sampler（被 flash_attn 阻断） |
| `models/qwen3.py` | 391 | Qwen3ForCausalLM（GQA + Q/K Norm + RoPE + SwiGLU） | 完整 | **qwen3.py:132-163 大段注释掉的 naive attention（死代码）**；`forward` 返回 logits（无 compute_logits）；`from_pretrained` 有 `# TODO: 完整映射`（:384） | test_Day2::test_qwen3_model / test_gqa（预期失败，见 §3.4） |
| `utils/context.py` | 80 | 全局 Context 单例 | 完整 | 无占位；`set_context(context)` 单对象签名 | test_Day1::test_context（坏，见 §3） |
| `utils/loader.py` | 129 | safetensors → 融合权重加载 | 完整 | 无占位；`load_model` + `load_model_weights` 别名 | 无直接测试（端到端依赖它） |

**占位标记全仓库扫描结论**：真正的 `TODO` 只有 2 处（`linear.py:231`、`qwen3.py:384`），均为"未来扩展"性质，不影响当前单卡功能。**无 `pass` 占位、无 `NotImplementedError`、无 `raise NotImplemented`**。死代码 1 处（`qwen3.py:132-163` 注释块）。

---

## 5. 历史分析校验结果（Phase 4）

### 5.0 历史脚手架清单（Phase 0 摘要）

| 文档 | 产出时间 | 核心结论 | 待办 |
|------|---------|---------|------|
| `.codex-plans/index.md` | 2026-04-23（git） | 当前无焦点任务；`repair-plans-docs` 已完成 | 无 |
| `.codex-plans/repair-plans-docs/{plan,findings,progress}.md` | 2026-04-23 | **修的是 `plans/` 教学文档**（不是代码）：补 00/02A、重写 04~07、统一文风 | progress 标记已全部完成 |
| `.codex-plans/day7-teaching-plans/` | — | **空目录，git 无任何历史** | — |
| `.codex-plans/project-status-analysis/` | — | **空目录，git 无任何历史**（本报告即写入此目录的首份文件） | — |
| `REVIEW_REPORT.md`（未跟踪） | 2026-06-11（mtime） | 评审 `experiments/` Day0~7 指南质量 B+；列 6 个测试 bug；与上游 nano-vllm 架构差异表 | 8 条改进建议 |
| `todo_list.md` | 2026-01-02起 | 7 天冲刺清单；正文 Day1~5 勾选 ✅ | 进度表却全标 ⬜ 未开始 |
| `docs/superpowers/plans/2026-04-30-*.md`（2 份） | 2026-04-30 | plans 11~13 主线重写计划 + Day13 文档整理计划 | 实施计划，checkbox 未勾 |

### 5.1 `repair-plans-docs/` —— 结论基本成立，但需澄清范围

- **依然成立**：它声称重写了 `plans/00`、新增 `02A`、重写 `04~07`。git log 证实 2026-04-22~23 确有 `97f5f9c`/`29c7699`/`f0df845` 三个 "plans: 重写…" 提交，`bbead9c`（04-30）补全 08~13 与 `.codex-plans`。这些文档确实存在（即今 `plans_archive/`）。
- **需要澄清（避免误读）**：`repair-plans-docs` 是**文档重构任务**，不是代码修复任务。它的"findings"（如"04-07 默认假设 01-03 已完成，导致和当前 HEAD 错位"）描述的是**文档与代码的错位**，而它给出的"修复"是改文档措辞，**并未改任何代码**。因此不能把它的"已完成"理解为"代码问题已修复"。
- **过时之处**：`progress.md` 提到"已检查 `plans/` 下绝对本地链接，当前没有发现坏链"——但 `plans/` 目录现已**整体重命名为 `plans_archive/` 且未提交**（`git status`显示 `plans/*.md` 全为 `D`，`plans_archive/` 为 `??`）。原文档内的跨篇链接（如01 篇结尾"下一篇：02-…"）指向的相对路径在重命名后是否仍有效，本次未逐一验证，列为待查。

### 5.2 `REVIEW_REPORT.md` —— 评审对象错位，多处与当前代码不符

**核心判定：REVIEW_REPORT 评审的是 `experiments/` 指南里的"完整代码块"（目标态），不是当前仓库代码（实际态）。**证据：

| REVIEW_REPORT 声称 | 当前仓库实测 | 结论 |
|--------------------|------------|------|
| `sampling_params.py`（59 行）含 top_k/top_p | 21 行，无 top_k/top_p | ❌ 不符 |
| `sampler.py`（186 行）支持 top_k/top_p | 95 行，forward 不收 top_k/top_p | ❌ 不符 |
| `llm_engine.py`（248 行） | 191 行 | ❌ 不符 |
| `qwen3.py`（434 行），`forward` 返回 hidden states + `compute_logits` 分离 | 391 行，`forward` 返回 logits，无 compute_logits | ❌ 不符 |
| `scheduler.py`（218 行） | 209 行 | ❌ 不符 |
| `model_runner.py`（381 行） | 367 行 | ❌ 不符 |
| `utils/loader.py`（130 行） | 129 行 | ✅ 接近 |

**依然成立的部分**（与当前代码核对一致）：
- 6 个测试 bug 汇总表（§3 已逐一实测/静态复核）：`set_context` 裸 kwargs、`Config(model=...)`、`attn(..., attention_mask=None)`、`store_kvcache` 5 参 vs 4 参、test_Day4 硬编码路径——**这些 bug 在当前仓库测试文件里确实存在**。
- 与上游 nano-vllm 的接口差异（`set_context(context)` 单对象、`store_kvcache` 合并 kv_cache、`Scheduler(config, block_manager)`、`postprocess` 2 参、`Sequence.block_size` 硬编码 256）——**与当前代码一致**。

**过时/误判的部分**：
- REVIEW_REPORT 说"所有 bug 均已在对应指南中被正确识别并给出修复方案"——但**修复只在 `experiments/` 指南文档里，仓库里的 `tests/test_Day*.py` 至今未改**（实测仍失败）。"指南识别了" ≠ "代码修复了"。
- REVIEW_REPORT 第七节"现有代码是 Day0 的初始状态"——这一判断对当前仓库**基本成立**（代码确实停留在主干完成、增强未做的状态），但它把 `experiments/` 指南的"完整代码"当成了"读者替换后能跑"的目标，与仓库实际无关。

### 5.3 `todo_list.md` —— 内部自相矛盾，且整体过时

- 正文 Day1~Day5 的 checkbox 大量勾选 ✅（如 1.7~1.10、2.x、3.x、4.x、5.x），Day6/Day7 未勾。
- 但文末"进度追踪"表（todo_list.md:393-401）**7 天全标 ⬜ 未开始**。
- 两者矛盾。结合代码实测：Day1~Day5 对应的**代码确实写了**（主干完整），但"跑通单卡 demo"（Day5 里程碑 5.7）**未经证实**（依赖 flash_attn，端到端未跑通）；Day6（TP/CUDA Graph）、Day7（Benchmark）**确实未做**（与 §2 矩阵 ⬜ 一致）。
- 结论：todo_list 的"正文勾选"高估了完成度（把"写了代码"当成"完成"），"进度表全 ⬜"又低估了（无视已写的主干）。**两者都不可单独采信**。

### 5.4 git 历史佐证："计划是否被执行过"

- **代码层面**：最后一次功能性代码提交是 `b0390c4`（2026-01-21 "适配计算图录制后的清除操作"）。此后所有提交（2026-03-09、04-22~30）**全部是文档**（"新增Day7学习教案"、"plans: 重写…"）。
- **工作区未提交改动**：`config.py`、`layers/linear.py`、`models/qwen3.py` 有未提交修改，但 `git diff` 显示**仅是行尾换行符/末尾空行**，无功能变化。即：**自 2026-01-21 以来，代码没有任何实质改动**。
- **推论**：`repair-plans-docs/`（04-23）和 `REVIEW_REPORT.md`（06-11）提出的所有"代码修复建议"（dtype 对齐、compute_logits、top_k/top_p、修测试 bug 等）**从未被落地到代码**——这与 §2 矩阵"01~04 增强未落地"完全吻合。`plans/`→`plans_archive/` 的重命名也是未提交的手工操作。

---

## 6. 建议的下一步（仅优先级，不含具体修复方案）

> 本轮只诊断。以下供下一轮任务决策参考，按"解锁后续工作的杠杆"排序。

1. **先解决 `layers/__init__.py` 的 eager import（最高杠杆）。**
   它一个文件就阻断了 test_Day2/test_Day3 整体收集 + test_Day4 两个用例。只要让 `layers.sampler`/`layers.layernorm` 等纯 torch 模块能脱离 flash_attn 独立导入，**无需装 flash_attn就能让可实测用例从 7 个涨到 ~14 个**，立刻暴露更多真实问题。这是"让测试套件能真正跑起来"的前置。

2. **修掉 4 处测试/实现接口错位（真 bug，与 flash_attn 无关）。**
   `set_context` 裸 kwargs（test_Day1/test_Day3）、`Config(model=...)`（test_Day1）、`attn(..., attention_mask=None)`（test_Day2）、`store_kvcache` 5 参 vs 4 参（test_Day3）。这些是"代码写了但测试从来没真正通过"的直接证据，修完才能让"主干已实现"这个判断有测试背书。

3. **~~把 flash_attn 装好（或给 attention.py 加 PyTorch fallback），再跑端到端 `example.py`。~~（已完成）**
   端到端已于 2026-07-25 实测通过（见 §3.4），主干"单卡能跑通"从代码阅读推断升级为实测确认。test_Day2::test_qwen3_model 那个无 Context 时 `None[layer_idx]` 的运行时 bug 也已实测确认（属测试文件问题，非主干问题）。

4. **统一并冻结"哪套文档算数"。**
   当前 `plans_archive/`（16 篇）、`experiments/`（Day0~7）、`REVIEW_REPORT.md`、`todo_list.md` 四套叙述互相打架。建议下一轮先明确：以哪套为唯一事实来源、`plans/`→`plans_archive/` 重命名是否要提交、`REVIEW_REPORT.md` 标注为"评审 experiments 指南、非当前代码"。否则后续任何"文档质量审计"都会继续踩这个分叉。

5. **进阶特性（05~13）整体未开始，若要推进，05(TP)/06(CUDA Graph) 依赖 04 的 `run_model()` 拆分。**
   即 04 篇的"增强改动"是 05/06 的前置。若目标是继续往进阶走，应先补 01~04 的增强（dtype 对齐、compute_logits、top_k/top_p、run_model 拆分），再谈 TP/CUDA Graph；若目标只是"让单卡主干真正跑通并被测试覆盖"，则 1~3 项足够，进阶可暂缓。

---

## 附：本次盘点未覆盖 / 待复测项

- ~~flash_attn 装好后重跑测试复核 §3.4 预判~~ → **已完成**：改用 `nano_vllm` 环境（自带 flash_attn 2.8.3）实测，§3.2/3.3 为真实结果，原静态预判全部得到证实。
- ~~端到端 `example.py` 仍未实际运行~~ → **已完成**：2026-07-25 在 `nano_vllm` 环境实测通过，两个 prompt 均生成正确文本（见 §3.4）。
- `plans_archive/`内部跨篇相对链接在 `plans/`→`plans_archive/` 重命名后是否仍有效，未逐一验证。
- `experiments/` Day0~7 八篇指南本身的内容质量已另文审计，见 `tutorial-audit-experiments-20260725.md`。
