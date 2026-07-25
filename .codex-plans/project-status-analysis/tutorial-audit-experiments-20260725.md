# 教程文档审计报告：`experiments/` (Day0–Day7) — 2026-07-25

> 审计对象：`experiments/` 下 8 篇新版指南（Day0~Day7，共 ~7250 行）。
> 审计标准（按用户要求）：**不修改任何代码**。对每个真实存在的代码问题，检查教程是否①指出了原因、②提出了修复方案；并整体评估教程的**全面性**与**和现有代码的匹配性**。
> 所有判断均以 Phase 1/2 已核实的真实代码与实测结果为基准（见 `current-status-20260725.md`）。
> 历史结论（`REVIEW_REPORT.md` 等）只作线索，已逐条重新核对。

---

## 1. 总体结论

**`experiments/` 是一套质量很高、与当前代码高度匹配的教学指南。** 核心判断：

1. **匹配性强**：每篇的"🔍 已有代码回顾"段落经逐条核对，**几乎都与当前仓库代码一致**——指南确实是在描述"当前真实代码"，而不是凭空想象或照抄上游。
2. **问题诊断真实可靠**：指南"⚠️ 问题分析"里列出的问题，**经核对全部真实存在**，且都给出了原因解释 + 具体修复方案。其中 Day3 §3 问题 4（`block_manager.py:298` off-by-one）是一个**连第一轮代码盘点都漏掉的真实正确性 bug**，指南却准确抓到了。
3. **测试 bug 覆盖到位**：6 个已知测试文件 bug，指南覆盖了 **5 个**，且修复方案都可用（与实测失败原因完全吻合）。
4. **全面性的主要缺口**：指南**完全没有提到 `layers/__init__.py` 的 eager import 问题**——这是当前"测试套件跑不起来"的最大单一原因（见下 §4）。

**一句话**：指南"指出原因 + 提出修复"这件事做得很扎实；短板在于①个别测试 bug 漏诊（Day2 漏 1 个）、②一处内部不一致（Day1）、③漏掉了导入组织这个系统性问题。

---

## 2. 逐篇审计

每篇按四个维度评：① 已有代码回顾是否匹配真实代码；② 问题分析是否真实+有原因+有修复；③ 完整代码/验证是否与真实接口一致；④ 全面性缺口。

### Day0 — 重温准备与架构总览（纯概念，329 行）

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | §5"当前代码状态"表列 10 项待完善点，**逐条核对全部属实**（Config 无 property / 无 top_k/top_p / 无 dtype 对齐 / forward 返回 logits / 无 reset_context / prefill 统计含 cached / block_manager:298 off-by-one / context.py 类型注解 / qwen3.py:380 `mode_path` 笔误 / layernorm docstring 拼写）。目录树与真实结构一致。 |
| 问题诊断 | §5 把 10 个问题分别指向对应 Day 指南，**指向准确**。§6 测试 bug 表（4 个文件）与实测失败原因吻合。 |
| 缺口 | 🟡 **轻度夸大**：把 `example.py` 标"✅ 可运行"、多个模块标"✅"，但这些**从未经测试证实**（测试套件实际跑不通，端到端依赖 flash_attn 未验证）。应标"代码已写、未验证"。 |

### Day1 — 数据结构层（879 行）

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | §2 四个文件（config/sampling_params/sequence/context）的现状描述**全部与真实代码一致**。 |
| 问题诊断 | §3 五个问题**全部真实**，原因+修复齐全：①Config 无 property（model_runner 走后门 `getattr`）②`temperature > 1e-10` 拒绝 greedy ③无 top_k/top_p ④Sequence 不复制 top_k/top_p ⑤context.py `max_context_len: int = None` 类型注解错。 |
| 验证 | §5 **准确诊断 test_Day1.py 两个 bug**（`set_context` 裸 kwargs、`Config(model=...)`），修复方案与实测 `TypeError` 完全对应，可用。 |
| 缺口 | 🟡 **内部不一致**：§3 问题 5 指出 context.py 类型注解 bug 并给了修复，但 §4.4 贴出的"完整代码"**仍保留 `max_context_len: int = None`**，且标注"不需要改"。即"指出的修复"没落进"完整代码"。（属静态类型问题，不影响运行，但作为教学材料自相矛盾。） |

### Day2 — 模型组件层（1140 行）

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | RMSNorm / SwiGLU / RoPE / 融合 Linear 四节**均与真实代码一致**。 |
| 问题诊断 | RMSNorm：诚实说明"无功能问题"，只指出 docstring 拼写（`redisual`/`normalized_putput`，layernorm.py:82,85，**属实**）。Linear：3 个问题真实——①`QKVLinear` 只给 weight 绑 loader 没绑 bias（linear.py:69，属实）②`default_weight_loader` 无 dtype/device 对齐（linear.py:238，属实）③建议 `torch.compile`。修复（`copy_weight_to_param` 辅助函数）具体可用。 |
| 验证 | §5 诊断 `test_gqa` 的 `attention_mask=None` bug（test_Day2.py:242），修复可用。 |
| 缺口 | 🔴 **漏诊 1 个测试 bug（已实测确认）**：test_Day2.py 还有 `test_qwen3_model`（:199）——直接 `model(input_ids)` **未设置 Context**，默认 `is_prefill=False` 会走 `_decode_attention`，执行 `context.kv_cache[layer_idx]`而 `kv_cache is None`。在 `nano_vllm` 环境实测：`TypeError: 'NoneType' object is not subscriptable`（attention.py:197），脚本在此崩溃，`test_gqa` 根本没机会执行。Day2 只覆盖了 test_Day2.py 两个 bug 中的 1 个。 |

### Day3 — PagedAttention 引擎（1124 行）★ 最强

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | Block / BlockManager 三方法 / Triton kernel / Attention 类**全部与真实代码一致**。 |
| 问题诊断 | §3 四个问题：1-3 诚实标注"不影响正确性"（stride/`.contiguous()`、fp16 转换开销、`_recover_block` 的 O(n) `list.remove`）；**问题 4 是真实正确性 bug**——`block_manager.py:298` `len(block_table) > 2` 应为 `>= 2`，破坏恰好 2 块序列的 Prefix Cache链式哈希。**指南标 🔴、给了影响分析和修复，诊断完全正确**。这个 bug 第一轮代码盘点漏掉了，指南抓到了。 |
| 验证 | §5 诊断 test_Day3.py 两个 bug：`set_context` 裸 kwargs 的修复**可用**；但 `store_kvcache` 的修复方案**有隐藏错误**（见缺口）。 |
| 缺口 | 🟡 §4"完整代码"**保留了问题 4 的 buggy 实现**，注明"读者可自行修复"。透明但未给出修复后的完整文件，读者需自己改。<br>🔴 **§5 的 `store_kvcache` 修复方案不可用**：指南建议 `kv_cache = torch.stack([k_cache, v_cache], dim=0)` 再传给 4 参 `store_kvcache`。但 `torch.stack` 会**复制**出新张量，kernel 写入的是副本，测试后面从原 `k_cache`/`v_cache` 验证会读到全零 → 断言失败。正确做法是直接按真实 `allocate_kv_cache` 的布局 `[2, num_blocks, block_size, num_kv_heads, head_dim]` 创建合并 cache，再从 `kv_cache[0]`/`kv_cache[1]` 验证（本次修 test_Day3.py 时已按此实现并跑通）。 |

### Day4 — Qwen3 模型与权重加载（983 行）

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | Qwen3Attention六步流程、loader 映射协议**与真实代码一致**。 |
| 问题诊断 | §3 四个问题**全部真实**：①forward 返回 logits（qwen3.py:369）②ModelRunner 依赖它 ③注释掉的 naive attention 死代码（qwen3.py:132-163）④`from_pretrained(cls, mode_path)` 笔误（qwen3.py:380）。原因+修复齐全。 |
| 完整代码 | §4 **真正把修复落进了完整代码**（forward→hidden states + `compute_logits()`、删死代码）——比 Day3"留 bug 给读者"更彻底。 |
| 验证 | §5 诊断 test_Day4.py 硬编码路径 bug（:3，`/home/psx/...`），给了 `os.path` 修复 + 补 `import os`，可用。接口验证脚本（hidden states / compute_logits 形状）设计合理。 |
| 缺口 | 无明显缺口。Day4 是"诊断+修复+验证"闭环最完整的一篇。 |

### Day5 — 调度器与 ModelRunner（1168 行）

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | Scheduler / ModelRunner 结构与真实代码一致。 |
| 问题诊断 | §4.2 三个问题真实：①`run()` 无 `reset_context()`（model_runner.py:319-362）②prefill token 统计含 cached（llm_engine.py:123）③Sampler 调用缺 top_k/top_p（指向 Day6）。 |
| 跨篇依赖 | §5 修复表**显式管理跨 Day 依赖**："run_model() 需同步 Day4 的 compute_logits 改动；若跳过 Day4 需先补"。这是多日教程最容易断链的地方，Day5 处理得好。 |
| 验证 | §7 交叉引用 Day3 的测试修复 + Context 清理手测 + `example.py` 端到端。 |
| 缺口 | 无明显缺口。 |

### Day6 — 推理链路（876 行）

| 维度 | 结论 |
|------|------|
| 代码回顾匹配 | §2 四个模块**全部准确**，且抓住了一个微妙点：Sampler 内部已处理 temperature=0，但 SamplingParams 仍拒绝它——指南两边都标了。 |
| 问题诊断 | §3 问题表**严重度分级合理**：top_k/top_p 三处缺失标 🔴 高、temperature=0 被拒标 🟡 中、prefill 统计标 🟡 中（注明"不影响推理正确性"）。 |
| 完整代码 | §4 给出 sampling_params / sampler / llm_engine / example 的完善版（新增 top_k/top_p 过滤）。 |
| 验证 | §5 Sampler 单测覆盖旧/新接口、greedy、`top_k=0` 边界，加端到端。设计扎实。 |
| 缺口 | 无明显缺口。 |

### Day7 — 进阶优化：CUDA Graph 与 Tensor Parallel（751 行）

| 维度 | 结论 |
|------|------|
| 定位 | **新增特性篇**（不是修当前代码的 bug），诚实声明是"教学版"。 |
| 前置条件 | §2.1 **明确声明依赖 Day4/Day5 的改动**（forward 返回 hidden states、compute_logits、run_model 拆出）。当前代码缺这些 → Day7 代码无法直接套在 HEAD 上，但指南对此透明。 |
| 全面性 | §4.3 对比表**准确标注**已实现/未实现（Chunked Prefill / Radix / Speculative / MoE / FP8 标 ❌）。§4.4 列下一步方向。 |
| 衔接 | §3.6 + §5 的"衔接说明"框**主动解决了 TP↔现有 Linear 的断链**：明确指出别名替换不能直接用（weight_loader 签名缺 `shard_id`），给两个方案，并建议单卡学习保留原 Linear。**这正是 `REVIEW_REPORT.md` 当年批评旧版 Day7 的点，新版已修复。** |
| 缺口 | 无实质缺口；唯一限制是它依赖未落地的 Day4/Day5 改动（属教程顺序问题，非文档质量问题）。 |

---

## 3. "真实代码问题 → 教程是否覆盖"对照表

以 Phase 1/2 核实的真实问题为行，检查教程覆盖度：

| # | 真实代码问题 | 证据 | 哪篇覆盖 | 指出原因? | 提出修复? |
|---|------------|------|---------|----------|----------|
| 1 | Config 无 property，下游散落 `getattr` | config.py 只有 `model` 别名 | Day0§5 / Day1§3-1 | ✅ | ✅ |
| 2 | SamplingParams 无 top_k/top_p | sampling_params.py 21 行 | Day0§5 / Day1§3-3 / Day6§3 | ✅ | ✅ |
| 3 | `temperature > 1e-10` 拒绝 greedy | sampling_params.py:21 | Day1§3-2 / Day6§2.3 | ✅ | ✅ |
| 4 | Sequence 不复制 top_k/top_p | sequence.py:95-97 | Day1§3-4 / Day6§2.4 | ✅ | ✅ |
| 5 | context.py 类型注解 `int = None` | context.py:49-50 | Day0§5 / Day1§3-5 | ✅ | ⚠️ 给了修复但 §4.4 完整代码未应用 |
| 6 | weight_loader 无 dtype/device 对齐 | linear.py:110,238 | Day0§5 / Day2§2.4 | ✅ | ✅ |
| 7 | QKVLinear.bias 未绑 loader | linear.py:69 | Day2§2.4 | ✅ | ✅ |
| 8 | forward 返回 logits，无 compute_logits | qwen3.py:369 | Day0§5 / Day4§3-1 | ✅ | ✅（§4 已应用） |
| 9 | qwen3.py 注释掉的死代码 | qwen3.py:132-163 | Day4§3-3 | ✅ | ✅（§4 已删除） |
| 10 | `from_pretrained(mode_path)` 笔误 | qwen3.py:380 | Day0§5 / Day4§3-4 | ✅ | ✅ |
| 11 | layernorm docstring 拼写 | layernorm.py:82,85 | Day0§5 / Day2§2.1 | ✅ | ✅（诚实标注为装饰性） |
| 12 | **block_manager.py:298 off-by-one**（正确性 bug） | block_manager.py:298 | Day0§5 / Day3§3-4 🔴 | ✅ | ✅（但 §4 完整代码保留 bug） |
| 13 | ModelRunner.run() 无 reset_context | model_runner.py:319-362 | Day0§5 / Day5§4.2-1 | ✅ | ✅ |
| 14 | prefill token 统计含 cached | llm_engine.py:123 | Day0§5 / Day5§4.2-2 / Day6§2.2 | ✅ | ✅ |
| 15 | Sampler docstring 声称 top_k/top_p 但无实现 | sampler.py:7-9 vs :28 | Day6§2.1（指出无 top_k/top_p） | ✅ | ✅（Day6 补实现） |
| 16 | **`layers/__init__.py` eager import 绑死 flash_attn** | layers/__init__.py:4 | **❌ 无任何一篇提及** | — | — |
| 17 | test_Day1 `set_context` 裸 kwargs | test_Day1.py:115 | Day1§5 | ✅ | ✅ |
| 18 | test_Day1 `Config(model=...)` | test_Day1.py:160 | Day1§5 | ✅ | ✅ |
| 19 | test_Day2 `attn(..., attention_mask=None)` | test_Day2.py:242 | Day2§5 | ✅ | ✅ |
| 20 | **test_Day2 `test_qwen3_model` 无 Context**（实测 `TypeError: 'NoneType' object is not subscriptable`） | test_Day2.py:199 | **❌ Day2 漏诊** | — | — |
| 21 | test_Day3 `set_context` 裸 kwargs | test_Day3.py:219 | Day3§5 | ✅ | ✅ |
| 22 | test_Day3 `store_kvcache` 5 参 vs 4 参 | test_Day3.py:262 | Day3§5 | ✅ | ⚠️ 给了修复但方案不可用（`torch.stack` 复制张量致验证失败，见 Day3 缺口） |
| 23 | test_Day4 硬编码绝对路径 | test_Day4.py:3 | Day4§5 | ✅ | ✅ |

**覆盖统计**：23 个真实问题中，教程覆盖 **21 个**（91%），其中 19 个"原因+修复"齐全可用。
- **2 个缺口**：#16（`layers/__init__.py` 导入组织，系统性问题，全教程未提）、#20（test_Day2 第二个 bug，Day2 漏诊）。
- **2 个修复瑕疵**：#5（Day1 指出 context.py 修复但完整代码未应用）、#22（Day3 的 `store_kvcache` 修复方案 `torch.stack`不可用）。

---

## 4. 全面性评估：教程漏掉了什么

### 4.1 最大缺口：`layers/__init__.py` 的 eager import（#16）

`layers/__init__.py:4` 在包导入时就 `from .attention import ...`，触发 `import flash_attn`。后果（Phase 2 实测）：
- 没装 flash_attn 时，连 `from layers.sampler import Sampler`、`from layers.layernorm import RMSNorm` 这种**纯 torch 模块都导入失败**；
- test_Day2 / test_Day3 **整体无法收集**；test_Day4 的 `test_linear_layers`/`test_sampler` 直接 `ModuleNotFoundError`。

**这是"测试套件跑不起来"的最大单一原因，且与代码逻辑无关。** 8 篇指南没有任何一篇提到它。Day2 讲 `layers/` 组件、Day3讲 attention，是最该提的地方，但都只讲功能、没讲"这个 `__init__.py` 会让纯 torch 测试也起不来"。

> 建议：Day2 或 Day0 应补一句——`layers/__init__.py` eager import attention 导致 flash_attn 成为整个 `layers` 包的硬依赖；若想让纯 torch 组件（RMSNorm/SiLU/RoPE/Sampler）能独立测试，应改为惰性导入或不在 `__init__.py` 里导入 attention。

### 4.2 次要缺口：test_Day2 的 `test_qwen3_model`（#20）

Day2 §5 只修了 `test_gqa` 的 `attention_mask` bug，漏了 `test_qwen3_model`（test_Day2.py:199）直接 `model(input_ids)` 不设 Context 的问题。即便修了 `attention_mask`、装好 flash_attn，这个用例仍会在 `_decode_attention` 里 `None[layer_idx]` 崩。Day2 应补上。

### 4.3 系统性观察：指南描述的是"应用全部修复后"的状态，而当前 HEAD 是"修复前"

每篇的"完整代码"都是修复后的版本，但**当前仓库代码停留在所有修复之前**（Phase 4 已证：自 2026-01-21 代码零实质改动）。这意味着：
- 指南的"已有代码回顾"匹配当前 HEAD（✅）；
- 但指南的"验证步骤"里那些"跑 example.py / 跑测试看是否通过"，**在读者真正按指南改完代码之前都不会通过**。
- 指南整体是"自洽的修复路线图"，但不是"对当前 HEAD 跑测试就能验证"的状态。这一点 Day0 应更显式地说明（目前 §5 把多个模块标"✅ 可运行"，容易让读者误以为当前就能跑通）。

---

## 5. 与 `REVIEW_REPORT.md` 的关系（历史校验）

`REVIEW_REPORT.md`（2026-06-11）评审的正是这套 `experiments/` 指南。核对后：

- **REVIEW_REPORT 对指南质量的正面评价基本成立**：结构统一、代码块完整无省略、bug 诊断准确。本审计独立复现了这些结论。
- **REVIEW_REPORT 批评旧版 Day7 "TP 衔接不清"——新版 Day7 已修复**（§3.6 + §5 衔接说明框）。说明 `experiments/` 是在吸收 REVIEW_REPORT 反馈后迭代的。
- **但 REVIEW_REPORT 有一处根本性误导**：它把指南里的"完整代码"（修复后目标态）当成了"当前仓库代码"来报告行号（如称 `sampling_params.py` 59 行带 top_k/top_p、`qwen3.py` 434 行 forward 返回 hidden states）。**这些与当前仓库不符**（实际 21 行无 top_k/top_p、391 行返回 logits）。因此 REVIEW_REPORT 的"代码完整性检查"小节不能当作当前代码状态的证据——它描述的是指南的目标态。本审计已严格区分"指南说什么"与"代码实际是什么"。
- **REVIEW_REPORT 也漏了 #16（`layers/__init__.py`）和 #20（test_Day2 第二个 bug）**——这两个缺口是本审计新发现的。

---

## 6. 结论与建议（仅诊断，不改代码/文档）

### 教程质量结论

`experiments/` 是一套**全面性与代码匹配性都相当高**的指南：
- ✅ "已有代码回顾"逐条匹配真实代码；
- ✅ 23 个真实问题覆盖 21 个，"指出原因 + 提出修复"做得扎实，含一个第一轮盘点漏掉的正确性 bug（block_manager:298）；
- ✅ 跨 Day 依赖（Day4→Day5→Day7）有显式管理；
- ✅ Day7 主动修复了 REVIEW_REPORT 当年指出的 TP 衔接缺口。

### 待改进项（按优先级，供下一轮"文档修订"参考）

1. **补 `layers/__init__.py` eager import 的说明**（最高优先）：这是测试跑不起来的最大原因，8 篇全漏。建议在 Day2（讲 layers 组件）或 Day0（讲环境/跑测试）补一段：解释它让 flash_attn 成为整包硬依赖、并给出惰性导入的修复方向。
2. **Day2 补 test_Day2.py 的第二个 bug**（`test_qwen3_model` 无 Context → `None[layer_idx]`），目前只修了 `attention_mask` 那个。
3. **Day1 消除内部不一致**：§3 问题 5 指出 context.py 类型注解修复，但 §4.4 完整代码应同步应用（把 `int = None` 改成 `int | None = None`），而不是标"不需要改"。
4. **Day0 §5 修正过度乐观的"✅可运行"标注**：明确区分"代码已写"与"已被测试/端到端证实"，避免读者误以为当前 HEAD 能直接跑通。
5. **Day3 §4 可考虑给出 block_manager:298 修复后的完整代码**，而非留 bug 让读者自行修复（当前虽透明，但读者易漏改）。
6. **Day3 §5 修正 `store_kvcache` 的修复方案**：把 `torch.stack([k_cache, v_cache])` 改为直接创建合并布局 `[2, num_blocks, block_size, num_kv_heads, head_dim]` 的 kv_cache 并从 `kv_cache[0]/[1]` 验证——现方案因 `stack` 复制张量会导致读者验证失败。

### 本轮边界

- **未修改任何主要学习代码、未修改任何教程文档**；经用户授权**修复了 4 个测试文件**（test_Day1~Day4.py）的接口 bug，修复后 21 个用例全部通过（详见 `current-status-20260725.md` §3.2）。
- 修 test_Day3.py::test_store_kvcache 时发现 Day3 §5 的修复方案不可用（见上 #22 / 待改进项6），已按正确布局实现。
- test_Day2 `test_qwen3_model`（#20，指南漏诊）的 `TypeError: 'NoneType' object is not subscriptable` 已实测确认并修复（补 prefill Context + CUDA/bf16）。
