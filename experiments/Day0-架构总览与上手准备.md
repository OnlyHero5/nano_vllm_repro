# Day 0 — 架构总览与上手准备

## 本篇定位

本篇不写一行代码，只做一件事：**把整个引擎的骨架和数据流铺开**，让你在动手改任何一层之前，先知道自己站在哪里。

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

## 2. 项目目录结构

```
nano_vll_repro/                  # 项目根目录
│
├── config.py                    # 全局配置（模型路径、显存利用率、块大小）
├── sampling_params.py           # 采样参数（temperature、max_tokens）
├── llm.py                       # 对外接口（LLM 类，就是 LLMEngine 的别名）
├── example.py                   # 端到端推理示例脚本
│
├── engine/                      # 运行时核心 — 整个推理引擎的大脑
│   ├── sequence.py              #   请求的运行时状态（token列表、状态机、block_table）
│   ├── block_manager.py         #   KV Cache 物理块管理器 + Prefix Cache
│   ├── scheduler.py             #   Continuous Batching 调度器（waiting/running 双队列）
│   ├── model_runner.py          #   把一批 Sequence 整理成模型输入，执行推理
│   └── llm_engine.py            #   顶层引擎循环（串联 Scheduler + ModelRunner）
│
├── layers/                      # 模型组件 — Transformer 的积木块
│   ├── linear.py                #   融合 Linear（QKVLinear / MergedLinear / RowLinear）
│   ├── layernorm.py             #   RMSNorm（含残差融合版本）
│   ├── activation.py            #   SwiGLU 激活函数（SiLU × gate）
│   ├── rotary_embedding.py      #   RoPE 旋转位置编码
│   ├── attention.py             #   PagedAttention（Triton store kernel + FlashAttention）
│   └── sampler.py               #   采样器（Greedy / Temperature / Gumbel-Max）
│
├── models/                      # 模型定义
│   ├── qwen3.py                 #   Qwen3 模型（GQA + Q/K Norm + 融合权重映射）
│   └── Qwen3-0.6B/              #   模型权重文件（需自行下载）
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.json
│
├── utils/                       # 工具
│   ├── context.py               #   全局 Context（在 ModelRunner 和 Attention 之间传元数据）
│   └── loader.py                #   权重加载器（safetensors → 融合权重映射）
│
├── tests/                       # 测试
│   ├── test_Day1.py             #   基础数据结构测试
│   ├── test_Day2.py             #   模型组件测试
│   ├── test_Day3.py             #   PagedAttention 测试
│   └── test_Day4.py             #   端到端测试
│
└── experiments/                 # 本实验指南（你正在读的）
    ├── Day0-架构总览与上手准备.md
    ├── Day1-数据结构层.md
    ├── ...（Day2-Day6：主线，逐层读透并改进）
    ├── Day7-进阶优化与总结.md
    ├── ...（Day8-Day13：进阶专题）
    └── Day13-CPU-KV-Block-Offload.md
```

> **两卷的读法不同**：
>
> - **Day0-Day6 是主线**：一层一层读透现有实现，找出它的薄弱处，动手补上。每篇改完都能用 `example.py` 和 `tests/test_Day1-4.py` 立刻验证。
> - **Day7-Day13 是进阶专题**：CUDA Graph、Chunked Prefill、Radix Cache、投机解码、MoE、量化、Offload。这几篇给的是完整设计与代码，但需要你亲手落地到自己的仓库，再按各篇"验收命令"一节验证。其中 Day11A 的测试与 demo 在纯 CPU 上就能跑通（预期输出见该篇 §8/§9）。
> - 进阶篇之间有依赖：Day7 依赖 Day4/5；Day8 依赖 Day4/5/6；Day10 依赖 Day4/8；Day11A 依赖 Day11。各篇开头有"前置依赖"框。

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

传统做法给每个请求预留一整块连续显存放 KV Cache，会产生内部/外部碎片，且相同前缀无法共享。PagedAttention 把 KV Cache 切成固定大小的「页」（block = 256 tokens），按需分配、允许不连续，相同前缀的请求还能共享物理页（Prefix Cache）。每条序列用 `block_table` 记录逻辑块到物理页的映射，写 cache 时再换算成逐 token 的 `slot_mapping`。这里先记住这个大方向即可，碎片问题的具体形态和两个映射表的细节，Day3 会展开。

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

## 5. 这一版实现的家底：哪些立住了，哪些还薄

这套代码能跑通端到端推理，骨架是结实的。但"能跑"和"经得起推敲"之间还有距离——主线六篇要走的，就是这段距离。先看清家底：

| 模块 | 文件 | 成色 |
|------|------|------|
| Config 配置 | `config.py` | 结实，基础功能完整 |
| SamplingParams | `sampling_params.py` | 还薄：只支持 temperature，缺 top_k/top_p |
| Sequence 状态机 | `engine/sequence.py` | 结实 |
| Block + BlockManager | `engine/block_manager.py` | 结实，Prefix Cache 已实现 |
| Scheduler | `engine/scheduler.py` | 结实，双队列调度已实现 |
| ModelRunner | `engine/model_runner.py` | 还薄：缺 `reset_context()` |
| LLMEngine | `engine/llm_engine.py` | 还薄：prefill token 统计不准 |
| RMSNorm | `layers/layernorm.py` | 结实，含残差融合 |
| SwiGLU 激活 | `layers/activation.py` | 结实 |
| RoPE | `layers/rotary_embedding.py` | 结实 |
| Attention + Triton kernel | `layers/attention.py` | 结实 |
| 融合 Linear | `layers/linear.py` | 还薄：weight_loader 的 dtype 对齐缺位 |
| Sampler | `layers/sampler.py` | 还薄：只支持 temperature |
| Qwen3 模型 | `models/qwen3.py` | 还薄：forward 直接返回 logits |
| 权重加载 | `utils/loader.py` | 结实 |
| Context | `utils/context.py` | 结实 |
| 对外接口 | `llm.py` | 结实 |
| 推理脚本 | `example.py` | 结实，可运行 |

### 主线要补的八处（Day1-Day6 逐个动手）

八处里没有一处是"随手改改"——每一处背后都有一个值得想清楚的设计问题，正文会连着问题一起讲：

1. **Config 缺少 property**：下游代码散落着 `getattr(hf_config, ...)`，配置读取没有单一出口
2. **SamplingParams 不支持 top_k/top_p**：采样策略被锁在 temperature 一个旋钮上
3. **Linear 的 weight_loader 缺乏 dtype/device 对齐**：safetensors 是 fp32、模型是 bf16 时会静默出错
4. **Qwen3ForCausalLM.forward() 直接返回 logits**：挡住了后面的 CUDA Graph 捕获
5. **ModelRunner.run() 没有 reset_context()**：全局 Context 会在 step 之间泄漏
6. **LLMEngine.step() 的 prefill token 统计**：把命中缓存的 prefix token 也算成了新算的
7. **`block_manager.py` 的 Off-by-One**：Prefix Cache 链式哈希条件 `len(block_table) > 2` 应为 `>= 2`——一个字符的差别，让哈希链丢掉第一个 block（Day3 §3）
8. **`context.py` 的类型注解**：`max_context_len: int = None` 应为 `int | None = None`（Day1 §3）

### 进阶扩展（Day7 会涉及）

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

> **`tests/` 下的测试文件可以直接运行。** 这些测试有几个容易写错的地方，对应 Day 指南的「验证步骤」给出了错误写法与修正写法的对照讲解，建议配合阅读：
>
> | 测试文件 | 易错点 | 对照讲解位置 |
> |---------|---------|------------|
> | `test_Day1.py` | `set_context()` 传参方式错误、`Config(model=...)` 参数名错误 | Day1 指南 §5 |
> | `test_Day2.py` | `attn()` 多传 `attention_mask=None`；`test_qwen3_model` 未设 Context 崩在 `None[layer_idx]` | Day2 指南 §5 |
> | `test_Day3.py` | `set_context()` 传参方式错误、`store_kvcache()` 签名错误 | Day3 指南 §5 |
> | `test_Day4.py` | 硬编码绝对路径 | Day4 指南 §5 |
>
> 建议：先跑 `example.py`（端到端推理），确认模型加载和基本推理正常后，再跑各 Day 测试。

如果 `example.py` 能输出中文文本，说明环境 OK，可以继续 Day1。

---

## 7. 本指南的使用方式

每天的内容结构：

```
1. 知识点讲解    ← 先想清「为什么这样设计」
2. 这一层长什么样 ← 读透现有实现
3. 这一版的薄弱处 ← 找出经不起推敲的地方
4. 完整代码      ← 改进后的完整文件
5. 验证步骤      ← 确认改动正确
```

建议的阅读路径：

```
Day0（本篇）→ Day3（PagedAttention 核心，最重要）→ Day1 → Day2 → Day4 → Day5 → Day6 → Day7
```

**为什么先读 Day3？** 因为 PagedAttention 是整个项目存在的理由。理解了它，再回头看 Sequence、BlockManager、Scheduler 就豁然开朗。

---

下一篇：**Day1 — 数据结构层**（Config / SamplingParams / Sequence / Context，逐个读透与改进）
