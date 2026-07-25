# nano-vLLM 复现指南

从零复现 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，用最精简的代码理解 vLLM 的核心推理架构。

**核心技术**：PagedAttention · Continuous Batching · Prefix Cache · FlashAttention · Triton Kernel

**参考仓库**：

- 主参考：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- 模型参考：HuggingFace `Qwen3-0.6B`
- 论文参考：[vLLM (SOSP'23)](https://arxiv.org/abs/2309.06180)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[RoPE](https://arxiv.org/abs/2104.09864)

---

## 项目目录结构

```
nano_vll_repro/
├── config.py                  # 全局配置（模型路径、显存、块大小）
├── sampling_params.py         # 采样参数（temperature、max_tokens）
├── llm.py                     # 对外接口（LLMEngine 的别名）
├── example.py                 # 端到端推理示例
├── engine/
│   ├── sequence.py            # 序列状态机（WAITING → RUNNING → FINISHED）
│   ├── block_manager.py       # PagedAttention 核心：块分配、释放、Prefix Cache
│   ├── scheduler.py           # Continuous Batching 调度器（双队列 + preemption）
│   ├── model_runner.py        # 模型执行器（KV Cache 分配、输入准备、前向推理）
│   └── llm_engine.py          # 推理引擎（串联 Scheduler + ModelRunner）
├── layers/
│   ├── linear.py              # 融合 Linear（QKVLinear / MergedLinear / RowLinear）
│   ├── layernorm.py           # RMSNorm（含残差融合版本）
│   ├── activation.py          # SwiGLU 激活函数
│   ├── rotary_embedding.py    # RoPE 旋转位置编码
│   ├── attention.py           # PagedAttention（Triton store kernel + FlashAttention）
│   └── sampler.py             # 采样器（Greedy / Temperature / Gumbel-Max）
├── models/
│   └── qwen3.py               # Qwen3 模型实现（GQA + Q/K Norm + 融合权重映射）
├── utils/
│   ├── context.py             # 全局 Context（在 ModelRunner 和 Attention 之间传递元数据）
│   └── loader.py              # 权重加载器（safetensors → 融合权重映射）
├── tests/
│   ├── test_Day1.py           # 基础数据结构测试
│   ├── test_Day2.py           # 模型主干测试
│   ├── test_Day3.py           # PagedAttention 测试
│   └── test_Day4.py           # 融合 Linear 测试
└── experiments/               # 📖 分步实验指南（8 篇，推荐阅读顺序）
    ├── Day0-重温准备与架构总览.md
    ├── Day1-数据结构层.md
    ├── Day2-模型组件层.md
    ├── Day3-PagedAttention引擎.md
    ├── Day4-Qwen3模型与权重加载.md
    ├── Day5-调度器与ModelRunner.md
    ├── Day6-推理链路.md
    └── Day7-进阶优化与总结.md
```

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.10+, CUDA 11.7+, PyTorch 2.0+
pip install torch transformers flash-attn triton safetensors tqdm xxhash
```

### 2. 准备模型权重

```bash
# 方式一：从 HuggingFace 下载
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# 方式二：如果已有本地权重，确保路径为 models/Qwen3-0.6B/
ls models/Qwen3-0.6B/*.safetensors
```

### 3. 运行端到端推理

```bash
cd nano_vll_repro
python example.py
```

预期输出：

```
CUDA: NVIDIA GeForce RTX xxx
============================================================
nano vllm Test
============================================================
[LLMEngine] 加载 Tokenizer...
[LLMEngine] 初始化 ModelRunner...
[ModelRunner] 加载模型：.../models/Qwen3-0.6B
[Loader] 发现 N 个权重文件
[Loader] 完成：加载 XX 个权重，跳过 X 个
[LLMEngine] 初始化完成！
[LLMEngine] - KV Cache: XXXX 块

[问题] 你好，请用300字介绍一下你自己。
[回答] ...
```

### 4. 运行测试

```bash
# 按里程碑运行
python tests/test_Day1.py
python tests/test_Day2.py
python tests/test_Day3.py
python tests/test_Day4.py

# 或用 pytest
pip install pytest
pytest tests -q
```

---

## 架构总览

一个请求从 `LLM.generate()` 到最终生成 token 的完整数据流：

```
用户文本
  │
  ▼ tokenize
token_ids
  │
  ▼ 封装
Sequence (token_ids + status + block_table + sampling_params)
  │
  ▼ Scheduler.schedule()
┌─────────────────────────────────────────────────────┐
│  双队列调度：                                         │
│  waiting ──(Prefill)──▶ running ──(Decode)──▶ 完成   │
│  决策依据：KV Cache 够不够？batch token 数超没超？     │
└─────────────────────────────────────────────────────┘
  │
  ▼ BlockManager.allocate() / append_slot()
┌─────────────────────────────────────────────────────┐
│  PagedAttention 显存管理：                            │
│  逻辑块 → 物理块映射 (block_table)                    │
│  token → cache 槽位映射 (slot_mapping)                │
│  Prefix Cache：内容哈希 → 块复用                      │
└─────────────────────────────────────────────────────┘
  │
  ▼ ModelRunner.prepare_prefill() / prepare_decode()
┌─────────────────────────────────────────────────────┐
│  输入准备：                                           │
│  Prefill：拼接所有 token，构建 cu_seqlens              │
│  Decode：每序列取 1 个 token，构建 block_tables        │
│  → 设置全局 Context                                   │
└─────────────────────────────────────────────────────┘
  │
  ▼ Qwen3ForCausalLM.forward()
┌─────────────────────────────────────────────────────┐
│  模型前向：                                           │
│  Embedding → [DecoderLayer × N] → Norm → LM Head    │
│  每层：Norm → QKV投影 → Q/K Norm → RoPE             │
│        → PagedAttention → O投影 → Norm → MLP         │
│                                                       │
│  Attention 层从全局 Context 获取：                      │
│  - is_prefill → 选择 flash_attn_varlen / with_kvcache │
│  - slot_mapping → Triton kernel 写入 KV Cache         │
│  - block_tables → Decode 时从 Cache 读取 KV           │
└─────────────────────────────────────────────────────┘
  │
  ▼ Sampler.forward()
  │  Gumbel-Max Trick：logits → 温度缩放 → 采样
  │
  ▼ Scheduler.postprocess()
  │  追加 token → 检查 EOS / max_tokens → 释放完成序列
  │
  ▼ 循环直到所有序列完成
  │
  ▼ tokenizer.decode()
生成文本
```

### 核心设计模式

**1. 权重融合 + weight_loader 协议**

HuggingFace 权重是分离的（`q_proj`, `k_proj`, `v_proj`），本项目融合成 `qkv_proj`。每个参数绑定 `weight_loader` 方法，`loader.py` 只负责分发，不理解内部布局：

```python
# layers/linear.py
self.weight.weight_loader = self._weight_loader

# utils/loader.py
weight_loader(param, loaded_weight, shard_id)  # shard_id = "q" / "k" / "v"
```

**2. 全局 Context 模式**

`ModelRunner` 在准备输入时设置全局 `Context`，`Attention` 层通过 `get_context()` 获取。避免修改中间层的 `forward` 签名：

```python
# engine/model_runner.py
set_context(context)

# layers/attention.py
context = get_context()
```

**3. Triton KV Cache 写入**

自定义 Triton kernel `store_kvcache_kernel` 将 K/V 写入 KV Cache 的指定 slot，比 PyTorch 原生索引更高效：

```python
# layers/attention.py
store_kvcache_kernel[grid](...)  # 每个 program 处理一个 token
```

---

## 🧪 实验指南（experiments/）

本仓库配有 8 篇实验指南，帮助你在三个月断档后快速回忆并深入理解 nano-vLLM 的全部实现。

每篇指南包含：**知识点讲解 → 已有代码回顾 → 问题诊断 → 完整代码 → 验证步骤**。

### 推荐阅读顺序

```
Day0 重温准备与架构总览  ← 从这里开始，帮你快速回忆
 ↓
Day3 PagedAttention 引擎  ← 最重要的模块，先读！
 ↓
Day1 数据结构层           ← Config/SamplingParams/Sequence/Context
 ↓
Day2 模型组件层           ← RMSNorm/RoPE/融合Linear
 ↓
Day4 Qwen3 模型与权重加载
 ↓
Day5 调度器与 ModelRunner
 ↓
Day6 完整推理链路         ← 到这里 example.py 完整可跑
 ↓
Day7 进阶优化与总结       ← CUDA Graph / TP / 知识图谱
```

### 指南目录

| 篇目 | 文件 | 主题 | 关键收获 |
|:---:|------|------|---------|
| Day0 | `Day0-重温准备与架构总览.md` | 架构总览、数据流图 | 看清全局：请求从进入到输出的完整路径 |
| Day1 | `Day1-数据结构层.md` | Config / SamplingParams / Sequence / Context | 理解状态机、全局 Context 模式、PagedAttention 的 block_table 预留 |
| Day2 | `Day2-模型组件层.md` | RMSNorm / SwiGLU / RoPE / 融合 Linear | 掌握 weight_loader 协议、QKV 融合原理、RoPE 旋转数学 |
| Day3 | `Day3-PagedAttention引擎.md` | Block / BlockManager / Attention | **核心**：理解分页 KV Cache、Prefix Cache、Triton 写入、FlashAttention 双模式 |
| Day4 | `Day4-Qwen3模型与权重加载.md` | Qwen3 完整模型 + safetensors 加载 | GQA 原理、packed_modules_mapping 映射协议、forward/logits 分离 |
| Day5 | `Day5-调度器与ModelRunner.md` | Scheduler / ModelRunner | Continuous Batching 双队列、Prefill vs Decode 输入准备的差异 |
| Day6 | `Day6-推理链路.md` | Sampler / LLMEngine / example.py | Gumbel-Max 采样、generate 循环、tqdm 吞吐监控 |
| Day7 | `Day7-进阶优化与总结.md` | CUDA Graph / TP / 知识图谱 | 进阶方向、面试高频问题、项目知识图谱 |

### 当前代码已知问题（Day1-Day6 会修复）

1. **`SamplingParams` 不支持 `top_k` / `top_p`** — 在 Day1 修复
2. **`Linear.weight_loader` 缺乏 dtype/device 对齐** — 在 Day2 修复
3. **`Qwen3ForCausalLM.forward()` 直接返回 logits** — 在 Day4 拆分为 `forward() + compute_logits()`
4. **`ModelRunner.run()` 没有 `reset_context()`** — 在 Day5 修复
5. **`LLMEngine.step()` 的 prefill token 统计不准** — 在 Day6 修复

> **注意**：旧的 `plans/` 目录已移至 `plans_archive/`。新旧指南有不兼容的差异，请只跟随 `experiments/` 中的指南。

---

## 当前实现状态

### 已实现（单卡推理链路完整可用）

| 模块 | 文件 | 核心能力 |
|------|------|---------|
| Config | `config.py` | 模型路径、显存利用率、块大小配置 |
| SamplingParams | `sampling_params.py` | temperature + max_tokens（待扩充 top_k/top_p） |
| Sequence | `engine/sequence.py` | 状态机（WAITING→RUNNING→FINISHED）+ pickle 序列化 |
| BlockManager | `engine/block_manager.py` | PagedAttention 块管理 + Prefix Cache（xxhash 链式哈希） |
| Scheduler | `engine/scheduler.py` | Continuous Batching（双队列 + LRU preemption） |
| ModelRunner | `engine/model_runner.py` | KV Cache 分配 + Prefill/Decode 输入准备 |
| LLMEngine | `engine/llm_engine.py` | 完整推理循环（generate → step → postprocess） |
| RMSNorm | `layers/layernorm.py` | 含残差融合 + `torch.compile` |
| SwiGLU | `layers/activation.py` | SiluAndMul 激活 |
| RoPE | `layers/rotary_embedding.py` | 预计算 cos/sin 缓存 |
| Attention | `layers/attention.py` | Triton store kernel + FlashAttention 双模式 |
| Linear | `layers/linear.py` | QKV / Gate-Up 融合 + weight_loader 协议 |
| Sampler | `layers/sampler.py` | Greedy + Temperature + Gumbel-Max（待扩充 top_k/top_p） |
| Qwen3 | `models/qwen3.py` | GQA + Q/K Norm + 融合权重映射 |
| Loader | `utils/loader.py` | safetensors → 融合权重分发 |

### 实验指南覆盖的进阶主题（Day7）

- **CUDA Graph** — decode 阶段录制计算图，消除 kernel launch 开销
- **Tensor Parallel** — 多卡权重切分与 all_reduce 通信
- **Chunked Prefill** — 长 prompt 分块处理的调度策略

## 性能参考

> 以下数据来自上游 nano-vllm 在 RTX 4070 Laptop (8GB) 上的测试，Qwen3-0.6B。

| 推理引擎 | 输出 Tokens | 耗时 (s) | 吞吐量 (tok/s) |
|---------|-----------|---------|---------------|
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

---

## 参考资料

- [vLLM 论文 (SOSP'23)](https://arxiv.org/abs/2309.06180) — PagedAttention 原始设计
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) — IO-aware 注意力算法
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — 旋转位置编码
- [nano-vllm 源码](https://github.com/GeeeekExplorer/nano-vllm) — 本项目的主参考
- [Qwen3 模型](https://huggingface.co/Qwen/Qwen3-0.6B) — 测试用模型
