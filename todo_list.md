# nano-vLLM 复现项目 - 7天冲刺待办清单

## 📋 项目概览

- **项目目标**：从零复现 nano-vllm，深入理解 vLLM 核心架构
- **参考仓库**：<https://github.com/GeeeekExplorer/nano-vllm.git>
- **时间周期**：7 天（2026年1月2日 - 2026年1月8日）
- **核心技术点**：PagedAttention、KV Cache 管理、FlashAttention、CUDA Graph、Tensor Parallelism

> **本文件与 `experiments/` 指南的关系**：本文件是**当初的 7 天冲刺计划**，记录"代码有没有写出来"；
> `experiments/` 是**教案**，讲"写出来的代码好不好、还能怎么改"。两边的 Day 编号互不对应
> （本文件 Day 1-7 = 冲刺日程；教案 Day0-13 = 章节序号），引用时注意区分。
> 两边的对应关系见下方[「与教案的对应关系」](#与教案的对应关系)一节。

---

## 🎯 核心学习目标

1. **理解 vLLM 架构**：PagedAttention 和 KV Cache 管理机制
2. **掌握高性能推理**：FlashAttention 集成、CUDA Graph 优化
3. **分布式系统**：基于多进程的 Tensor Parallelism (TP)

---

## 📁 项目目录结构

下面是**当前实际的**目录结构（冲刺之初的目标树用的是 `nanovllm/` 这个包名，落地时改成了
仓库根目录直接铺开）。逐文件的职责说明见教案 `experiments/Day0-架构总览与上手准备.md` §2。

```
nano_vll_repro/
├── config.py
├── sampling_params.py
├── llm.py                  # 对外接口（LLMEngine 的别名）
├── example.py              # 端到端推理示例
├── engine/
│   ├── sequence.py
│   ├── block_manager.py
│   ├── scheduler.py
│   ├── model_runner.py
│   └── llm_engine.py
├── layers/
│   ├── layernorm.py
│   ├── activation.py
│   ├── rotary_embedding.py
│   ├── attention.py
│   ├── linear.py
│   └── sampler.py
├── models/
│   └── qwen3.py
├── utils/
│   ├── context.py
│   └── loader.py           # 权重加载（safetensors → 融合权重映射）
├── tests/                  # test_Day1-4.py，按里程碑组织
└── experiments/            # 教案（15 篇）
```

---

## 📅 每日任务清单

---

### Day 1: 基础设施与数据结构 (Infrastructure & Data Structures)

**学习目标**：搭建项目骨架，定义核心数据结构，理解 Sequence 抽象

#### 上午 (AM) - 环境准备与项目初始化

- [x] **1.1** 克隆参考仓库到本地，通读 README 和项目结构
- [x] **1.2** 搭建开发环境（Python 3.10+, PyTorch 2.0+, CUDA）
- [x] **1.3** 安装依赖：`flash-attn`, `triton`, `transformers`
- [x] **1.4** 创建项目目录结构（按上述目录树）
- [x] **1.5** 阅读参考仓库的 `config.py`，理解配置项含义
- [x] **1.6** 阅读参考仓库的 `sampling_params.py`，理解采样参数

#### 下午 (PM) - 核心数据结构实现

- [x] **1.7** 手写 `config.py`
  - [x] 定义 `Config` 类
  - [x] 处理模型路径、并发参数、dtype 等配置
- [x] **1.8** 手写 `sampling_params.py`
  - [x] 定义 `SamplingParams` 数据类
  - [x] 包含 temperature, top_k, top_p, max_tokens 等参数
- [x] **1.9** 手写 `engine/sequence.py`
  - [x] 定义 `SequenceStatus` 枚举（Waiting/Running/Finished）
  - [x] 定义 `Sequence` 类
  - [x] 实现 token_ids 管理
  - [x] 理解并预留 `block_table` 属性（PagedAttention 伏笔）
- [x] **1.10** 手写 `utils/context.py`
  - [x] 实现全局上下文管理器
  - [x] 用于跨模块传递元数据

#### Day 1 检查点 ✅

- [x] 能够创建 `Config` 实例并打印配置
- [x] 能够创建 `Sequence` 实例并管理状态
- [x] 理解 Request → Sequence 封装的设计意图
- [x] 理解 `block_table` 在后续 PagedAttention 中的作用

---

### Day 2: 模型架构搭建 (Model Architecture)

**学习目标**：实现 Transformer 核心组件，搭建 Qwen3/Llama 模型骨架

#### 上午 (AM) - 基础层实现

- [x] **2.1** 阅读 RMSNorm 论文/博客，理解与 LayerNorm 的区别
- [x] **2.2** 手写 `layers/layernorm.py`
  - [x] 实现 `RMSNorm` 类
  - [x] 注意 eps 参数和权重初始化
- [x] **2.3** 阅读 SiLU 激活函数原理
- [x] **2.4** 手写 `layers/activation.py`
  - [x] 实现 `SiluAndMul` 类（GLU 变体）
- [x] **2.5** 深入阅读 RoPE 论文，理解旋转位置编码数学原理
- [x] **2.6** 手写 `layers/rotary_embedding.py`
  - [x] 实现频率计算 (freqs)
  - [x] 实现 apply_rotary_emb 函数

#### 下午 (PM) - 模型骨架搭建

- [x] **2.7** 阅读 Qwen3/Llama 模型结构，理解各层组成
- [x] **2.8** 手写 `models/qwen3.py`（简化版）
  - [x] 实现 `Qwen3Attention` 类（先用普通 nn.Linear）
  - [x] 实现 `Qwen3MLP` 类
  - [x] 实现 `Qwen3DecoderLayer` 类
  - [x] 实现 `Qwen3Model` 类（Embedding + Layers + Norm）
  - [x] 实现 `Qwen3ForCausalLM` 类（加 lm_head）
- [x] **2.9** 编写简单测试：随机输入能否通过 forward

#### Day 2 检查点 ✅

- [x] RMSNorm 单元测试通过
- [x] RoPE 位置编码计算正确
- [x] 模型能完成一次 forward pass（随机权重）
- [x] 能清晰解释 RoPE 的数学原理

---

### Day 3: 核心灵魂 - 显存管理与 PagedAttention (Memory Management)

**学习目标**：理解并实现 vLLM 最核心的 KV Cache 分页管理

#### 上午 (AM) - Block 管理器

- [x] **3.1** 精读 vLLM PagedAttention 论文（重点第3节）
- [x] **3.2** 理解物理块 vs 逻辑块的概念
- [x] **3.3** 理解 block_table 映射机制
- [x] **3.4** 手写 `engine/block_manager.py`
  - [x] 定义 `Block` 类
    - [x] 包含 block_id, ref_count
    - [x] 实现 hash 计算（Prefix Caching 用）
  - [x] 定义 `BlockManager` 类
    - [x] 初始化 free_blocks 池
    - [x] 实现 `allocate()` 方法 - 为新序列分配块
    - [x] 实现 `append_slot()` 方法 - 追加 token 时的块管理

#### 下午 (PM) - Attention 层与 KV Cache

- [x] **3.5** 阅读 FlashAttention 论文，理解其优化原理
- [x] **3.6** 学习 flash_attn 库 API
- [x] **3.7** 手写 `layers/attention.py`
  - [x] 集成 `flash_attn` 库
  - [x] 实现 `Attention` 类
  - [x] 实现 KV Cache 的读写逻辑
  - [x] 编写 `store_kvcache` 函数（Triton/PyTorch）
- [x] **3.8** 理解 Prefill vs Decode 阶段的 Attention 差异
- [x] **3.9** 画图：物理块、逻辑块、block_table 的关系

#### Day 3 检查点 ✅

- [x] 能清晰解释 PagedAttention 解决了什么问题
- [x] BlockManager 能正确分配和释放块
- [x] Attention 层能正确读写 KV Cache
- [x] 理解 hash 在 Prefix Caching 中的作用

---

### Day 4: 调度器与执行引擎 (Scheduler & Execution)

**学习目标**：实现 Continuous Batching 调度逻辑

#### 上午 (AM) - 调度器实现

- [x] **4.1** 阅读 vLLM 论文中的调度策略部分
- [x] **4.2** 理解 Continuous Batching vs Static Batching
- [x] **4.3** 手写 `engine/scheduler.py`
  - [x] 维护 `waiting` 队列
  - [x] 维护 `running` 队列
  - [x] 实现 `add_sequence()` 方法
  - [x] 实现 `schedule()` 方法
    - [x] 检查显存是否足够
    - [x] 将 waiting 序列移到 running
    - [x] 调用 BlockManager 分配块
  - [x] 实现 `postprocess()` 方法
    - [x] 处理已完成的序列
    - [x] 释放对应的块

#### 下午 (PM) - ModelRunner 基础版

- [x] **4.4** 手写 `engine/model_runner.py`（基础版）
  - [x] 实现 `__init__` - 加载模型和 tokenizer
  - [x] 实现 `allocate_kv_cache()` - 预分配 GPU 显存
  - [x] 实现 `prepare_input()` 方法
    - [x] 将多个 Sequence 的 token 拼成 batch
    - [x] 生成 attention_mask
    - [x] 生成 position_ids
    - [x] 构建 block_tables tensor
  - [x] 实现基础 `run()` 方法 - 执行 forward
- [x] **4.5** 理解调度器如何与 BlockManager 交互

#### Day 4 检查点 ✅

- [x] 调度器能正确管理 waiting/running 队列
- [x] 能根据显存情况做出调度决策
- [x] ModelRunner 能正确准备 batch 输入
- [x] 理解 Continuous Batching 的优势

---

### Day 5: 完整推理循环与采样 (Inference Loop & Sampler)

**学习目标**：串联所有组件，实现完整 generate 流程

#### 上午 (AM) - Sampler 实现

- [x] **5.1** 复习采样算法：Greedy, Temperature, Top-K, Top-P
- [x] **5.2** 手写 `layers/sampler.py`
  - [x] 实现温度缩放
  - [x] 实现 Top-K 过滤
  - [x] 实现 Top-P (Nucleus) 过滤
  - [x] 实现最终采样逻辑
  - [x] 实现 `Sampler` 类整合以上功能
- [x] **5.3** 编写 Sampler 单元测试

#### 下午 (PM) - LLMEngine 与推理循环

- [x] **5.4** 手写 `engine/llm_engine.py`
  - [x] 实现 `__init__` - 初始化各组件
  - [x] 实现 `add_request()` - 添加推理请求
  - [x] 实现 `step()` 函数
    - [x] 调用 `scheduler.schedule()`
    - [x] 区分 Prefill 和 Decode 阶段
    - [x] 调用 `model_runner.run()`
    - [x] 调用 `sampler` 采样
    - [x] 调用 `scheduler.postprocess()`
  - [x] 实现 `generate()` - 完整生成循环
- [x] **5.5** 手写 `llm.py`
  - [x] 实现用户侧 API
  - [x] 封装 LLMEngine
- [x] **5.6** 编写 `example.py` 测试脚本
- [x] **5.7** 🎉 **里程碑**：跑通单卡推理 demo！

#### Day 5 检查点 ✅

- [x] Sampler 采样结果符合预期分布
- [x] 能够完成一个完整的文本生成
- [x] Prefill 和 Decode 阶段正确区分
- [x] example.py 能正常运行并输出结果

---

### Day 6: 高级特性 - 张量并行与 CUDA Graph (Optimization)

**学习目标**：实现多卡并行和 CUDA Graph 优化

> 这一天的两个主题，教案 `Day7-进阶优化与总结.md` 已给出完整设计与代码（§1-2 CUDA Graph、
> §3 TP），照着落地即可。注意 TP 版 `layers/linear.py` 是 Day2 版本的**整体替换**，
> 单卡模式下行为等价——细节见该篇的衔接说明。

#### 上午 (AM) - Tensor Parallelism

- [ ] **6.1** 学习 Tensor Parallelism 原理（Megatron-LM 论文）
- [ ] **6.2** 理解 ColumnParallel vs RowParallel 的区别
- [ ] **6.3** 学习 `torch.distributed` API（init_process_group, all_reduce）
- [ ] **6.4** 手写 `layers/linear.py`
  - [ ] 实现 `ColumnParallelLinear` 类
    - [ ] 权重按列切分
    - [ ] forward 后无需 all_reduce
  - [ ] 实现 `RowParallelLinear` 类
    - [ ] 权重按行切分
    - [ ] forward 后需要 all_reduce
- [ ] **6.5** 修改 `qwen3.py`，替换为并行 Linear 层

#### 下午 (PM) - CUDA Graph 优化

- [ ] **6.6** 学习 CUDA Graph 原理和使用场景
- [ ] **6.7** 理解为什么 CUDA Graph 对 Decode 阶段有效
- [ ] **6.8** 修改 `engine/model_runner.py`（进阶版）
  - [ ] 添加多进程初始化代码
  - [ ] 实现 `capture_cudagraph()` 方法
    - [ ] 录制计算图
    - [ ] 处理静态 shape 要求
  - [ ] 实现 `replay()` 方法
    - [ ] 重放录制的计算图
  - [ ] 在 Decode 阶段使用 CUDA Graph
- [ ] **6.9** 测试多卡运行（如果有多卡）

#### Day 6 检查点 ✅

- [ ] ColumnParallel 和 RowParallel 正确切分权重
- [ ] all_reduce 通信正确
- [ ] CUDA Graph 录制和重放正常工作
- [ ] 能清晰解释 TP 的通信模式

---

### Day 7: 测试、Benchmark 与简历打磨 (Final Polish)

**学习目标**：验证性能，整理文档，转化为简历语言

> `readme.md` 已经写好（含架构图与数据流），7.6 只剩补性能数据。教案
> `Day7-进阶优化与总结.md` §4 的知识图谱可直接用于 7.5 的通读复盘。

#### 上午 (AM) - 性能测试与调试

- [ ] **7.1** 写 `bench.py` 并跑性能测试（仓库里还没有这个脚本）
  - [ ] 测量吞吐量（Tokens/s）
  - [ ] 测量首 token 延迟（Time to First Token）
  - [ ] 测量生成延迟（Time per Output Token）
- [ ] **7.2** 与 HuggingFace 原生实现对比性能
- [ ] **7.3** 检查内存泄漏
  - [ ] 使用 `torch.cuda.memory_stats()`
  - [ ] 长时间运行测试
- [ ] **7.4** 修复发现的 bug

#### 下午 (PM) - 代码复盘与简历整理

- [ ] **7.5** Code Review：通读所有代码
  - [ ] 重点复习 `block_manager.py`
  - [ ] 重点复习 `attention.py`
  - [ ] 重点复习 `scheduler.py`
  - [ ] 确保理解每一行代码
- [ ] **7.6** 整理项目文档
  - [ ] 编写 README.md
  - [ ] 添加架构图
  - [ ] 记录性能数据
- [ ] **7.7** 准备简历描述（见下方模板）
- [ ] **7.8** 准备面试可能被问到的问题

#### Day 7 检查点 ✅

- [ ] 性能数据记录完整
- [ ] 代码无明显 bug
- [ ] 能流畅解释任意模块的实现
- [ ] 简历描述准备完成

---

## 📝 简历亮点模板

### 项目名称

**nano-vLLM：高性能 LLM 推理引擎复现**

### 项目描述

从零实现了类 vLLM 的高性能大模型推理引擎，支持 Qwen3/Llama 系列模型。

### 技术亮点（根据实际完成情况选择）

- [ ] 实现了 **PagedAttention** 内存管理机制，通过分页管理 KV Cache 解决显存碎片化问题
- [ ] 实现了 **Continuous Batching** 调度策略，相比静态 Batching 提升 X% 吞吐量
- [ ] 集成 **FlashAttention**，优化 Attention 计算性能
- [ ] 实现了基于 **NCCL 的 Tensor Parallelism**，支持多卡推理
- [ ] 使用 **CUDA Graph** 优化 Decode 阶段，减少 Kernel Launch 开销
- [ ] 手写实现 **RoPE**、**RMSNorm** 等 Transformer 核心组件

### 性能数据（待填写）

| 指标 | 本项目 | HuggingFace | 提升 |
|------|--------|-------------|------|
| 吞吐量 (tokens/s) | - | - | - |
| 首 Token 延迟 | - | - | - |
| 显存占用 | - | - | - |

---

## ❓ 面试高频问题准备

### PagedAttention 相关

- [ ] Q: 什么是 PagedAttention？解决了什么问题？
- [ ] Q: 物理块和逻辑块是如何映射的？
- [ ] Q: Prefix Caching 是如何实现的？

### 调度相关

- [ ] Q: Continuous Batching 和 Static Batching 有什么区别？
- [ ] Q: 调度器是如何决定哪些序列可以运行的？

### 并行相关

- [ ] Q: ColumnParallel 和 RowParallel 的区别是什么？
- [ ] Q: 为什么 RowParallel 后需要 all_reduce？

### 优化相关

- [ ] Q: CUDA Graph 的原理是什么？为什么对 Decode 有效？
- [ ] Q: FlashAttention 是如何优化的？

---

## 📊 进度追踪

| Day | 上午 | 下午 | 状态 |
|-----|------|------|------|
| Day 1 | 环境 & 项目初始化 | 核心数据结构 | ✅ 已完成 |
| Day 2 | 基础层 (Norm/Activation/RoPE) | 模型骨架 | ✅ 已完成 |
| Day 3 | Block 管理器 | Attention & KV Cache | ✅ 已完成 |
| Day 4 | 调度器 | ModelRunner | ✅ 已完成 |
| Day 5 | Sampler | LLMEngine & 完整流程 | ✅ 已完成（单卡推理 demo 已跑通） |
| Day 6 | Tensor Parallelism | CUDA Graph | ⬜ 未开始 |
| Day 7 | 性能测试 | 代码复盘 & 简历 | ⬜ 未开始 |

**图例**：⬜ 未开始 | 🟡 进行中 | ✅ 已完成

> 注：此表与上方逐项清单同步——Day 1-5 的全部子项已勾选完成，Day 6（TP / CUDA Graph）与 Day 7（性能测试 / 复盘）的子项均未开始。

**「已完成」指代码写出来并跑通了，不代表这一层已经经得起推敲。** 教案主线（Day1-Day6）
逐层复查后列出 8 处待改进，见下节。

---

## 🔧 教案主线指出的待改进项（10 处）

代码能跑通端到端推理，骨架是结实的；这 10 处是「能跑」和「经得起推敲」之间的距离。
每处背后都有一个设计问题，展开讲解见对应教案篇目（汇总见 Day0 §5）。

- [ ] **F1** `Config` 缺少 property：下游散落着 `getattr(hf_config, ...)`，配置读取没有单一出口 —— 教案 Day1 §3
- [ ] **F2** `SamplingParams` 不支持 top_k/top_p：采样策略被锁在 temperature 一个旋钮上 —— 教案 Day1 §3
- [ ] **F3** `SamplingParams` 不允许 temperature=0：而这正是 greedy 解码，现在只能拿 1e-4 糊过去 —— 教案 Day1 §3
- [ ] **F4** `Sequence` 没有透传 top_k/top_p：参数补齐后还得让它真的流到采样器 —— 教案 Day1 §3
- [ ] **F5** `context.py` 的类型注解：`max_context_len: int = None` 应为 `int | None = None` —— 教案 Day1 §3
- [ ] **F6** `Linear.weight_loader` 缺 dtype/device 对齐：safetensors 是 fp32、模型是 bf16 时静默出错 —— 教案 Day2 §2.4
- [ ] **F7** `block_manager.py` 的 Off-by-One：链式哈希条件 `len(block_table) > 2` 应为 `>= 2`，一个字符让哈希链丢掉第一个 block —— 教案 Day3 §3
- [ ] **F8** `Qwen3ForCausalLM.forward()` 直接返回 logits：挡住后面的 CUDA Graph 捕获 —— 教案 Day4 §3
- [ ] **F9** `ModelRunner.run()` 没有 `reset_context()`：全局 Context 在 step 之间泄漏 —— 教案 Day5 §4.2
- [ ] **F10** `LLMEngine.step()` 的 prefill token 统计：把命中缓存的 prefix token 也算成新算的 —— 教案 Day6 §3

---

## 🗺️ 与教案的对应关系

本文件的冲刺日程与教案章节是两套编号，对应关系如下：

| 本文件 | 主题 | 对应教案篇目 |
|---|---|---|
| Day 1 | 环境 / 核心数据结构 | Day1 数据结构层 |
| Day 2 | 基础层 / 模型骨架 | Day2 模型组件层、Day4 Qwen3 与权重加载 |
| Day 3 | Block 管理器 / Attention | Day3 PagedAttention 引擎 |
| Day 4 | 调度器 / ModelRunner | Day5 调度器与 ModelRunner |
| Day 5 | Sampler / LLMEngine | Day6 推理链路 |
| Day 6 | Tensor Parallelism / CUDA Graph | Day7 进阶优化与总结 |
| Day 7 | 性能测试 / 复盘 | Day7 §4 知识图谱与总结、§5 验证命令 |

教案 Day8-Day13 是冲刺计划里没有的进阶专题，属于额外的扩展方向：

- Day8 Chunked Prefill · Day9 Radix Prefix Cache · Day10 投机解码
- Day11 / Day11A MoE 与 expert offloading · Day12 KV Cache int8 量化 · Day13 CPU KV swap

这几篇给的是完整设计与代码，需要亲手落地到源码，再按各篇「验收命令」验证。

---

## 📚 参考资料

- [vLLM 论文](https://arxiv.org/abs/2309.06180)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)
- [nano-vllm 源码](https://github.com/GeeeekExplorer/nano-vllm)

---

*最后更新: 2026年7月26日（与 experiments/ 教案对齐：补待改进项、编号对应关系、实际目录结构）*
