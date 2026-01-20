# nano-vLLM 复现项目 - 7天冲刺待办清单

## 📋 项目概览

- **项目目标**：从零复现 nano-vllm，深入理解 vLLM 核心架构
- **参考仓库**：<https://github.com/GeeeekExplorer/nano-vllm.git>
- **时间周期**：7 天（2026年1月2日 - 2026年1月8日）
- **核心技术点**：PagedAttention、KV Cache 管理、FlashAttention、CUDA Graph、Tensor Parallelism

---

## 🎯 核心学习目标

1. **理解 vLLM 架构**：PagedAttention 和 KV Cache 管理机制
2. **掌握高性能推理**：FlashAttention 集成、CUDA Graph 优化
3. **分布式系统**：基于多进程的 Tensor Parallelism (TP)

---

## 📁 项目目录结构（目标）

```
nanovllm/
├── __init__.py
├── config.py
├── sampling_params.py
├── llm.py
├── engine/
│   ├── __init__.py
│   ├── sequence.py
│   ├── block_manager.py
│   ├── scheduler.py
│   ├── model_runner.py
│   └── llm_engine.py
├── layers/
│   ├── __init__.py
│   ├── layernorm.py
│   ├── activation.py
│   ├── rotary_embedding.py
│   ├── attention.py
│   ├── linear.py
│   └── sampler.py
├── models/
│   ├── __init__.py
│   └── qwen3.py
└── utils/
    ├── __init__.py
    └── context.py
```

---

## 📅 每日任务清单

---

### Day 1: 基础设施与数据结构 (Infrastructure & Data Structures)

**学习目标**：搭建项目骨架，定义核心数据结构，理解 Sequence 抽象

#### 上午 (AM) - 环境准备与项目初始化

- [✅️] **1.1** 克隆参考仓库到本地，通读 README 和项目结构
- [✅️] **1.2** 搭建开发环境（Python 3.10+, PyTorch 2.0+, CUDA）
- [✅️] **1.3** 安装依赖：`flash-attn`, `triton`, `transformers`
- [✅️] **1.4** 创建项目目录结构（按上述目录树）
- [✅️] **1.5** 阅读参考仓库的 `config.py`，理解配置项含义
- [✅️] **1.6** 阅读参考仓库的 `sampling_params.py`，理解采样参数

#### 下午 (PM) - 核心数据结构实现

- [✅️] **1.7** 手写 `nanovllm/config.py`
  - [✅️] 定义 `Config` 类
  - [✅️] 处理模型路径、并发参数、dtype 等配置
- [✅️] **1.8** 手写 `nanovllm/sampling_params.py`
  - [✅️] 定义 `SamplingParams` 数据类
  - [✅️] 包含 temperature, top_k, top_p, max_tokens 等参数
- [✅️] **1.9** 手写 `nanovllm/engine/sequence.py`
  - [✅️] 定义 `SequenceStatus` 枚举（Waiting/Running/Finished）
  - [✅️] 定义 `Sequence` 类
  - [✅️] 实现 token_ids 管理
  - [✅️] 理解并预留 `block_table` 属性（PagedAttention 伏笔）
- [✅️] **1.10** 手写 `nanovllm/utils/context.py`
  - [✅️] 实现全局上下文管理器
  - [✅️] 用于跨模块传递元数据

#### Day 1 检查点 ✅

- [✅️] 能够创建 `Config` 实例并打印配置
- [✅️] 能够创建 `Sequence` 实例并管理状态
- [✅️] 理解 Request → Sequence 封装的设计意图
- [✅️] 理解 `block_table` 在后续 PagedAttention 中的作用

---

### Day 2: 模型架构搭建 (Model Architecture)

**学习目标**：实现 Transformer 核心组件，搭建 Qwen3/Llama 模型骨架

#### 上午 (AM) - 基础层实现

- [✅️ ] **2.1** 阅读 RMSNorm 论文/博客，理解与 LayerNorm 的区别
- [ ✅️] **2.2** 手写 `nanovllm/layers/layernorm.py`
  - [ ✅️] 实现 `RMSNorm` 类
  - [✅️ ] 注意 eps 参数和权重初始化
- [ ✅️] **2.3** 阅读 SiLU 激活函数原理
- [ ✅️] **2.4** 手写 `nanovllm/layers/activation.py`
  - [ ✅️] 实现 `SiluAndMul` 类（GLU 变体）
- [ ✅️] **2.5** 深入阅读 RoPE 论文，理解旋转位置编码数学原理
- [ ✅️] **2.6** 手写 `nanovllm/layers/rotary_embedding.py`
  - [ ✅️] 实现频率计算 (freqs)
  - [ ✅️] 实现 apply_rotary_emb 函数

#### 下午 (PM) - 模型骨架搭建

- [ ✅️] **2.7** 阅读 Qwen3/Llama 模型结构，理解各层组成
- [ ✅️] **2.8** 手写 `nanovllm/models/qwen3.py`（简化版）
  - [✅️ ] 实现 `Qwen3Attention` 类（先用普通 nn.Linear）
  - [ ✅️] 实现 `Qwen3MLP` 类
  - [✅️ ] 实现 `Qwen3DecoderLayer` 类
  - [✅️ ] 实现 `Qwen3Model` 类（Embedding + Layers + Norm）
  - [ ✅️] 实现 `Qwen3ForCausalLM` 类（加 lm_head）
- [ ✅️] **2.9** 编写简单测试：随机输入能否通过 forward

#### Day 2 检查点 ✅

- [✅️ ] RMSNorm 单元测试通过
- [ ✅️] RoPE 位置编码计算正确
- [ ✅️] 模型能完成一次 forward pass（随机权重）
- [ ✅️] 能清晰解释 RoPE 的数学原理

---

### Day 3: 核心灵魂 - 显存管理与 PagedAttention (Memory Management)

**学习目标**：理解并实现 vLLM 最核心的 KV Cache 分页管理

#### 上午 (AM) - Block 管理器

- [✅️ ] **3.1** 精读 vLLM PagedAttention 论文（重点第3节）
- [✅️ ] **3.2** 理解物理块 vs 逻辑块的概念
- [ ✅️] **3.3** 理解 block_table 映射机制
- [ ✅️] **3.4** 手写 `nanovllm/engine/block_manager.py`
  - [ ✅️] 定义 `Block` 类
    - [ ✅️] 包含 block_id, ref_count
    - [✅️ ] 实现 hash 计算（Prefix Caching 用）
  - [✅️ ] 定义 `BlockManager` 类
    - [ ✅️] 初始化 free_blocks 池
    - [ ✅️] 实现 `allocate()` 方法 - 为新序列分配块
    - [ ✅️] 实现 `append_slot()` 方法 - 追加 token 时的块管理

#### 下午 (PM) - Attention 层与 KV Cache

- [ ✅️] **3.5** 阅读 FlashAttention 论文，理解其优化原理
- [✅️ ] **3.6** 学习 flash_attn 库 API
- [✅️ ] **3.7** 手写 `nanovllm/layers/attention.py`
  - [✅️ ] 集成 `flash_attn` 库
  - [✅️ ] 实现 `Attention` 类
  - [✅️ ] 实现 KV Cache 的读写逻辑
  - [✅️ ] 编写 `store_kvcache` 函数（Triton/PyTorch）
- [✅️ ] **3.8** 理解 Prefill vs Decode 阶段的 Attention 差异
- [✅️ ] **3.9** 画图：物理块、逻辑块、block_table 的关系

#### Day 3 检查点 ✅

- [✅️ ] 能清晰解释 PagedAttention 解决了什么问题
- [✅️ ] BlockManager 能正确分配和释放块
- [✅️ ] Attention 层能正确读写 KV Cache
- [ ✅️] 理解 hash 在 Prefix Caching 中的作用

---

### Day 4: 调度器与执行引擎 (Scheduler & Execution)

**学习目标**：实现 Continuous Batching 调度逻辑

#### 上午 (AM) - 调度器实现

- [ ✅️] **4.1** 阅读 vLLM 论文中的调度策略部分
- [ ✅️] **4.2** 理解 Continuous Batching vs Static Batching
- [ ✅️] **4.3** 手写 `nanovllm/engine/scheduler.py`
  - [ ✅️] 维护 `waiting` 队列
  - [ ✅️] 维护 `running` 队列
  - [ ✅️] 实现 `add_sequence()` 方法
  - [ ✅️] 实现 `schedule()` 方法
    - [ ✅️] 检查显存是否足够
    - [ ✅️] 将 waiting 序列移到 running
    - [ ]✅️ 调用 BlockManager 分配块
  - [ ✅️] 实现 `postprocess()` 方法
    - [ ✅️] 处理已完成的序列
    - [ ✅️] 释放对应的块

#### 下午 (PM) - ModelRunner 基础版

- [ ✅️] **4.4** 手写 `nanovllm/engine/model_runner.py`（基础版）
  - [ ✅️] 实现 `__init__` - 加载模型和 tokenizer
  - [ ✅️] 实现 `allocate_kv_cache()` - 预分配 GPU 显存
  - [ ✅️] 实现 `prepare_input()` 方法
    - [ ✅️] 将多个 Sequence 的 token 拼成 batch
    - [ ✅️] 生成 attention_mask
    - [ ✅️] 生成 position_ids
    - [ ✅️] 构建 block_tables tensor
  - [ ✅️] 实现基础 `run()` 方法 - 执行 forward
- [ ✅️] **4.5** 理解调度器如何与 BlockManager 交互

#### Day 4 检查点 ✅

- [ ✅️] 调度器能正确管理 waiting/running 队列
- [ ✅️] 能根据显存情况做出调度决策
- [ ✅️] ModelRunner 能正确准备 batch 输入
- [ ✅️] 理解 Continuous Batching 的优势

---

### Day 5: 完整推理循环与采样 (Inference Loop & Sampler)

**学习目标**：串联所有组件，实现完整 generate 流程

#### 上午 (AM) - Sampler 实现

- [ ✅️] **5.1** 复习采样算法：Greedy, Temperature, Top-K, Top-P
- [ ✅️] **5.2** 手写 `nanovllm/layers/sampler.py`
  - [ ✅️] 实现温度缩放
  - [ ✅️] 实现 Top-K 过滤
  - [ ✅️] 实现 Top-P (Nucleus) 过滤
  - [ ✅️] 实现最终采样逻辑
  - [ ✅️] 实现 `Sampler` 类整合以上功能
- [ ✅️] **5.3** 编写 Sampler 单元测试

#### 下午 (PM) - LLMEngine 与推理循环

- [ ✅️] **5.4** 手写 `nanovllm/engine/llm_engine.py`
  - [ ✅️] 实现 `__init__` - 初始化各组件
  - [ ✅️] 实现 `add_request()` - 添加推理请求
  - [ ✅️] 实现 `step()` 函数
    - [ ✅️] 调用 `scheduler.schedule()`
    - [ ✅️] 区分 Prefill 和 Decode 阶段
    - [ ✅️] 调用 `model_runner.run()`
    - [ ✅️] 调用 `sampler` 采样
    - [ ✅️] 调用 `scheduler.postprocess()`
  - [ ✅️] 实现 `generate()` - 完整生成循环
- [ ✅️] **5.5** 手写 `nanovllm/llm.py`
  - [ ✅️] 实现用户侧 API
  - [ ✅️] 封装 LLMEngine
- [ ✅️ **5.6** 编写 `example.py` 测试脚本
- [ ✅️] **5.7** 🎉 **里程碑**：跑通单卡推理 demo！

#### Day 5 检查点 ✅

- [ ✅️] Sampler 采样结果符合预期分布
- [ ✅️] 能够完成一个完整的文本生成
- [ ✅️] Prefill 和 Decode 阶段正确区分
- [ ✅️] example.py 能正常运行并输出结果

---

### Day 6: 高级特性 - 张量并行与 CUDA Graph (Optimization)

**学习目标**：实现多卡并行和 CUDA Graph 优化

#### 上午 (AM) - Tensor Parallelism

- [ ] **6.1** 学习 Tensor Parallelism 原理（Megatron-LM 论文）
- [ ] **6.2** 理解 ColumnParallel vs RowParallel 的区别
- [ ] **6.3** 学习 `torch.distributed` API（init_process_group, all_reduce）
- [ ] **6.4** 手写 `nanovllm/layers/linear.py`
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
- [ ] **6.8** 修改 `nanovllm/engine/model_runner.py`（进阶版）
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

#### 上午 (AM) - 性能测试与调试

- [ ] **7.1** 运行 `bench.py` 进行性能测试
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
| Day 1 | 环境 & 项目初始化 | 核心数据结构 | ⬜ 未开始 |
| Day 2 | 基础层 (Norm/Activation/RoPE) | 模型骨架 | ⬜ 未开始 |
| Day 3 | Block 管理器 | Attention & KV Cache | ⬜ 未开始 |
| Day 4 | 调度器 | ModelRunner | ⬜ 未开始 |
| Day 5 | Sampler | LLMEngine & 完整流程 | ⬜ 未开始 |
| Day 6 | Tensor Parallelism | CUDA Graph | ⬜ 未开始 |
| Day 7 | 性能测试 | 代码复盘 & 简历 | ⬜ 未开始 |

**图例**：⬜ 未开始 | 🟡 进行中 | ✅ 已完成

---

## 📚 参考资料

- [vLLM 论文](https://arxiv.org/abs/2309.06180)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)
- [nano-vllm 源码](https://github.com/GeeeekExplorer/nano-vllm)

---

*最后更新: 2026年1月2日*
