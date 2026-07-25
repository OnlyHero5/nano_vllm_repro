# Day 0 — 重温准备与架构总览

## 本篇定位

你已经三个月没碰这个项目了。本篇的任务是**帮你快速回忆起整个项目的骨架**，不写任何代码，纯复习。

读完本篇后，你应该能：
- 画出项目的数据流图
- 说出每个文件的职责
- 理解 PagedAttention、Continuous Batching、KV Cache 这三件事是如何协作的

---

## 1. 这个项目到底在做什么？

一句话：**用最精简的代码（~1200行），实现一个类似 vLLM 的大模型推理引擎。**

它的核心问题是：**如何让 GPU 高效地同时服务多个用户请求？**

打个比方：
- 传统做法是「一个用户占一整块显存，用完才释放」——像餐厅给每个客人一间包房
- nano-vLLM 的做法是「把显存切成小块，按需分配，共享前缀」——像自助餐，大家各取所需，共享公共区域

---

## 2. 项目目录结构（回忆版）

```
nano_vll_repro/                  # 项目根目录
│
├── config.py                    # 全局配置（模型路径、显存利用率、块大小）
├── sampling_params.py           # 采样参数（temperature、max_tokens）
├── llm.py                       # 对外接口（LLM 类，就是 LLMEngine 的别名）
├── example.py                   # 端到端推理示例脚本
│
├── engine/                      # 🧠 运行时核心 — 整个推理引擎的大脑
│   ├── sequence.py              #   请求的运行时状态（token列表、状态机、block_table）
│   ├── block_manager.py         #   KV Cache 物理块管理器 + Prefix Cache
│   ├── scheduler.py             #   Continuous Batching 调度器（waiting/running 双队列）
│   ├── model_runner.py          #   把一批 Sequence 整理成模型输入，执行推理
│   └── llm_engine.py            #   顶层引擎循环（串联 Scheduler + ModelRunner）
│
├── layers/                      # 🏗️ 模型组件 — Transformer 的积木块
│   ├── linear.py                #   融合 Linear（QKVLinear / MergedLinear / RowLinear）
│   ├── layernorm.py             #   RMSNorm（含残差融合版本）
│   ├── activation.py            #   SwiGLU 激活函数（SiLU × gate）
│   ├── rotary_embedding.py      #   RoPE 旋转位置编码
│   ├── attention.py             #   PagedAttention（Triton store kernel + FlashAttention）
│   └── sampler.py               #   采样器（Greedy / Temperature / Gumbel-Max）
│
├── models/                      # 🎯 模型定义
│   ├── qwen3.py                 #   Qwen3 模型（GQA + Q/K Norm + 融合权重映射）
│   └── Qwen3-0.6B/              #   模型权重文件（需自行下载）
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.json
│
├── utils/                       # 🔧 工具
│   ├── context.py               #   全局 Context（在 ModelRunner 和 Attention 之间传元数据）
│   └── loader.py                #   权重加载器（safetensors → 融合权重映射）
│
├── tests/                       # ✅ 测试
│   ├── test_Day1.py             #   基础数据结构测试
│   ├── test_Day2.py             #   模型组件测试
│   ├── test_Day3.py             #   PagedAttention 测试
│   └── test_Day4.py             #   端到端测试
│
└── experiments/                 # 📖 本实验指南（你正在读的）
    ├── Day0-重温准备与架构总览.md
    ├── Day1-数据结构层.md
    ├── ...
    └── Day7-进阶优化.md
```

---

## 3. 架构核心：一条请求的完整生命周期

这是理解整个项目最重要的一张图。先记个大概，后面每天会反复遇到：

```
用户输入: "你好，请介绍一下你自己"
        │
        ▼  tokenize
  [108386, 100168, 3837, ...]    ← token IDs
        │
        ▼  封装
  Sequence 对象被创建：
    · token_ids = [108386, 100168, ...]
    · status = WAITING              ← 等待被调度
    · block_table = []               ← 还没分 KV Cache
    · sampling_params = SamplingParams(temperature=0.7, max_tokens=64)
        │
        ▼  Scheduler.schedule()
  ┌─────────────────────────────────────────────────┐
  │ 调度器决定「这轮该算谁」:                         │
  │                                                  │
  │ 规则1: 优先处理 waiting 队列（新请求的 prefill）   │
  │   如果 KV Cache 不够 → 拒绝（等下一轮）            │
  │   如果 token budget 超了 → 拒绝                  │
  │                                                  │
  │ 规则2: waiting 空了 → 处理 running 队列（decode）  │
  │   如果显存不够 → 抢占（preempt）最慢的请求          │
  └─────────────────────────────────────────────────┘
        │
        ▼  BlockManager.allocate() / append_slot()
  ┌─────────────────────────────────────────────────┐
  │ PagedAttention 显存管理:                          │
  │                                                  │
  │ 物理内存被切成固定大小的「页」(block = 256 tokens) │
  │                                                  │
  │ 物理页池: [页0] [页1] [页2] [页3] ... [页N]       │
  │              ↓      ↓                           │
  │ 请求A的 block_table: [页0, 页2]    ← 不连续没关系  │
  │ 请求B的 block_table: [页1, 页3]                  │
  │                                                  │
  │ Prefix Cache: 如果请求A和B的前N个token相同，       │
  │ 它们可以共享物理页                                 │
  └─────────────────────────────────────────────────┘
        │
        ▼  ModelRunner.prepare_prefill() 或 prepare_decode()
  ┌─────────────────────────────────────────────────┐
  │ 把 Sequence 列表翻译成模型能吃的张量:               │
  │                                                  │
  │ Prefill: 所有 token 拼成一个大 batch              │
  │   input_ids:  [108386, 100168, 3837, ...]        │
  │   positions:  [0, 1, 2, ...]                     │
  │   cu_seqlens: [0, len_seq0, len_seq0+len_seq1]   │
  │   slot_mapping: [字节偏移量列表]                   │
  │                                                  │
  │ Decode: 每序列只取最后一个 token                    │
  │   input_ids:  [last_token_A, last_token_B]       │
  │   block_tables: [[页0, 页2], [页1, 页3]]  ← 查表  │
  │                                                  │
  │ → 设置全局 Context，Attention 层会去读取            │
  └─────────────────────────────────────────────────┘
        │
        ▼  Qwen3ForCausalLM.forward()
  ┌─────────────────────────────────────────────────┐
  │ 模型前向计算:                                     │
  │                                                  │
  │ Embedding → DecoderLayer × 28 → Norm → LM Head   │
  │                                                  │
  │ 每层 DecoderLayer:                                │
  │   RMSNorm → QKV 投影 → Q/K Norm → RoPE           │
  │          → Attention（读/写 KV Cache）→ O 投影     │
  │          → RMSNorm → SwiGLU MLP                  │
  │                                                  │
  │ Attention 层通过全局 Context 获取:                  │
  │   · slot_mapping  → Triton kernel 把 K/V 写入 cache │
  │   · block_tables  → FlashAttention 从 cache 读 K/V │
  │   · cu_seqlens    → Prefill 时的序列边界           │
  └─────────────────────────────────────────────────┘
        │
        ▼  Sampler.forward()
  ┌─────────────────────────────────────────────────┐
  │ logits → / temperature → softmax → Gumbel-Max 采样 │
  │ → 得到下一个 token ID                             │
  └─────────────────────────────────────────────────┘
        │
        ▼  Scheduler.postprocess()
  ┌─────────────────────────────────────────────────┐
  │ · 新 token 追加到 Sequence                       │
  │ · 如果是 EOS 或长度达到 max_tokens → FINISHED     │
  │ · 释放完成的序列的 KV Cache 页                     │
  └─────────────────────────────────────────────────┘
        │
        ▼  循环（回到 Scheduler.schedule()）
  ... 直到所有请求都 FINISHED
        │
        ▼  tokenizer.decode()
  生成的文本: "你好！我是 Qwen，一个由阿里云开发的大语言模型..."
```

---

## 4. 四个你最需要记住的核心概念

### 4.1 PagedAttention（分页注意力）

| 传统做法 | PagedAttention |
|---------|---------------|
| 每个请求分配一整块连续显存放 KV Cache | 把 KV Cache 切成固定大小的「页」(block = 256 tokens) |
| 请求A: [████████░░░░░░░░] 预留太多浪费 | 请求A: [页0]→[页2]→[页5] 按需分配，可以不连续 |
| 请求B: [████████████████] 放不下 | 请求B 可以复用请求A 的前几个页（Prefix Cache） |

核心数据结构：
- **block_table**（逻辑→物理映射）：`[17, 203, 41]` 表示逻辑块0→物理页17，逻辑块1→物理页203
- **slot_mapping**（token→槽位）：`slot = 物理页号 × block_size + 页内偏移`

### 4.2 Continuous Batching（连续批处理）

传统做法是「等整个 batch 的所有请求都完成，才开始下一批」。Continuous Batching 是：

> **不等整个 batch 完成，每个 step 都可能加入新请求。**

实现方式：维护两个队列
- `waiting`：新请求，还没开始算（Prefill 阶段把它们拉进来）
- `running`：已经在生成中的请求（Decode 阶段逐个推进）

### 4.3 KV Cache

Transformer 生成 token 时，每次都要看「之前所有 token」的 Key 和 Value。如果不缓存，每次都要重新计算整段历史 → O(n²) 复杂度。

KV Cache 的做法：
- Prefill 阶段：把整段 prompt 的 K/V 算出来，**存进 cache**
- Decode 阶段：只算新 token 的 K/V，**追加到 cache**，然后用 FlashAttention 从 cache 里高效读取历史

### 4.4 全局 Context 模式

模型有很多层（DecoderLayer），每层里面包着 Attention。如果要把 `slot_mapping`、`block_tables` 这些元数据层层传递下去，需要改所有中间函数的签名。

nano-vLLM 用了一个更轻量的方案：
- `ModelRunner` 在准备输入时调用 `set_context(...)`，把本轮元数据写入全局单例
- `Attention` 层里调用 `get_context()` 读取
- 每个 step 结束后调用 `reset_context()` 清理

这是一个典型的「跨层数据传递」模式，类似于 React 的 Context API。

---

## 5. 当前代码状态：哪些完成了，哪些还需要补

### ✅ 已经完成的（可正常运行）

| 模块 | 文件 | 状态 |
|------|------|------|
| Config 配置 | `config.py` | ✅ 基础功能完整 |
| SamplingParams | `sampling_params.py` | ⚠️ 只支持 temperature（缺 top_k/top_p） |
| Sequence 状态机 | `engine/sequence.py` | ✅ 完整 |
| Block + BlockManager | `engine/block_manager.py` | ✅ Prefix Cache 已实现 |
| Scheduler | `engine/scheduler.py` | ✅ 双队列调度已实现 |
| ModelRunner | `engine/model_runner.py` | ⚠️ 缺 `reset_context()` |
| LLMEngine | `engine/llm_engine.py` | ⚠️ prefill token 统计不准确 |
| RMSNorm | `layers/layernorm.py` | ✅ 含残差融合 |
| SwiGLU 激活 | `layers/activation.py` | ✅ |
| RoPE | `layers/rotary_embedding.py` | ✅ |
| Attention + Triton kernel | `layers/attention.py` | ✅ |
| 融合 Linear | `layers/linear.py` | ⚠️ weight_loader 有 dtype 问题 |
| Sampler | `layers/sampler.py` | ⚠️ 只支持 temperature |
| Qwen3 模型 | `models/qwen3.py` | ⚠️ forward 直接返回 logits |
| 权重加载 | `utils/loader.py` | ✅ |
| Context | `utils/context.py` | ✅ |
| 对外接口 | `llm.py` | ✅ |
| 推理脚本 | `example.py` | ✅ 可运行 |

### ⚠️ 需要完善的地方（本指南 Day1-Day6 会修复）

1. **Config 缺少 property**：下游代码散落着 `getattr(hf_config, ...)`，应该统一到 Config 的 property 里
2. **SamplingParams 不支持 top_k/top_p**：只能控制 temperature
3. **Linear 的 weight_loader 缺乏 dtype/device 对齐**：如果 safetensors 是 fp32、模型是 bf16，可能静默出错
4. **Qwen3ForCausalLM.forward() 直接返回 logits**：不方便后续做 CUDA Graph
5. **ModelRunner.run() 没有 reset_context()**：Context 可能在 step 之间泄漏
6. **LLMEngine.step() 的 prefill token 统计**：把已缓存的 prefix token 也算进去了
7. **`block_manager.py` 第 298 行 Off-by-One**：Prefix Cache 链式哈希条件 `len(block_table) > 2` 应为 `>= 2`（Day3 指南 §3 问题 4）
8. **`context.py` 类型注解错误**：`max_context_len: int = None` 应为 `int | None = None`（Day1 指南 §3 问题 5）
9. **`qwen3.py` 参数名笔误**：`from_pretrained(cls, mode_path)` 应为 `model_path`（Day4 指南 §3 问题 4）
10. **`layernorm.py` docstring 拼写**：`redisual` → `residual`，`normalized_putput` → `normalized_output`（Day2 指南 §2）

### 🔮 进阶扩展（Day7 会涉及）

- Tensor Parallel（多卡推理）
- CUDA Graph（Decode 加速）
- Chunked Prefill（长 prompt 分块处理）

---

## 6. 快速跑通验证（确认环境没问题）

```bash
cd nano_vll_repro

# 1. 确认模型权重在（约 1.2GB）
ls models/Qwen3-0.6B/model.safetensors

# 2. 如果还没有，下载
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# 3. 安装依赖
pip install torch transformers flash-attn triton safetensors tqdm xxhash pytest

# 4. 跑端到端推理
python example.py

# 5. 跑已有测试
python tests/test_Day1.py
python tests/test_Day2.py
python tests/test_Day3.py
python tests/test_Day4.py
```

> **⚠️ 当前测试文件有已知 bug，直接运行会报错。** 各 bug 的修复方法详见对应 Day 指南的「验证步骤」：
>
> | 测试文件 | 主要 bug | 修复说明位置 |
> |---------|---------|------------|
> | `test_Day1.py` | `set_context()` 传参方式错误、`Config(model=...)` 参数名错误 | Day1 指南 §5 |
> | `test_Day2.py` | `attn()` 多传了 `attention_mask=None` 参数 | Day2 指南 §5 |
> | `test_Day3.py` | `set_context()` 传参方式错误、`store_kvcache()` 签名错误 | Day3 指南 §5 |
> | `test_Day4.py` | 硬编码绝对路径 | Day4 指南 §5 |
>
> 建议：先跑 `example.py`（端到端推理），确认模型加载和基本推理正常后，再逐 Day 修复测试文件。

如果 `example.py` 能输出中文文本，说明环境 OK，可以继续 Day1。

---

## 7. 本指南的使用方式

每天的内容结构：

```
1. 📖 知识点讲解   ← 先理解「为什么」
2. 🔍 已有代码回顾  ← 帮你回忆三个月前写了什么
3. ⚠️  当前问题分析  ← 指出哪里需要改
4. 📝 完整代码      ← 可以复制粘贴的完整文件
5. ✅ 验证步骤      ← 确认改动正确
```

建议的阅读路径：

```
Day0（本篇）→ Day3（PagedAttention 核心，最重要）→ Day1 → Day2 → Day4 → Day5 → Day6 → Day7
```

**为什么先读 Day3？** 因为 PagedAttention 是整个项目存在的理由。理解了它，再回头看 Sequence、BlockManager、Scheduler 就豁然开朗。

---

下一篇：**Day1 — 数据结构层**（Config / SamplingParams / Sequence / Context 的已有代码回顾与完善）
