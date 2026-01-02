## 📋 项目概览

- **项目目标**：从零复现 nano-vLLM，深入理解 vLLM 核心架构
- **参考仓库**：<https://github.com/GeeeekExplorer/nano-vllm.git>
- **时间周期**：7 天（2026年1月2日 - 2026年1月8日）
- **核心技术点**：PagedAttention、KV Cache 管理、FlashAttention、CUDA Graph、Tensor Parallelism

---

## ✨ 项目特点

- **PagedAttention**：通过分页管理 KV Cache，解决显存碎片化问题。
- **高性能推理**：集成 FlashAttention 和 CUDA Graph，优化推理性能。
- **分布式支持**：基于 NCCL 的 Tensor Parallelism，支持多卡推理。
- **模块化设计**：从配置、数据结构到推理引擎，模块化实现，便于扩展。

---

## 📁 项目目录结构

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

## 🚀 快速开始

### 1. 环境准备

- **依赖环境**：
  - Python 3.10+
  - PyTorch 2.0+
  - CUDA 11.7+
- **安装依赖**：
  ```bash
  pip install flash-attn triton transformers
  ```

### 2. 克隆仓库

```bash
git clone https://github.com/OnlyHero5/nano_vll_repro.git
cd nano_vll_repro
```

### 3. 运行示例

运行 `example.py` 测试推理流程：

```bash
python example.py --model qwen3 --device cuda --max_tokens 128
```

---

## 🛠️ 开发路线图

### Day 1: 基础设施与数据结构

- [x] 创建项目目录结构
- [x] 实现 `Config` 和 `SamplingParams` 数据类
- [x] 实现 `Sequence` 数据结构，支持 token 管理

### Day 2: 模型架构搭建

- [x] 实现 Transformer 核心组件（RMSNorm、RoPE、Attention 等）
- [x] 搭建 Qwen3 模型骨架，支持随机输入的 forward pass

### Day 3: 显存管理与 PagedAttention

- [x] 实现 `BlockManager`，支持 KV Cache 分页管理
- [x] 集成 FlashAttention，优化 Attention 性能

### Day 4: 调度器与执行引擎

- [x] 实现 Continuous Batching 调度器
- [x] 实现 `ModelRunner`，支持批量推理

### Day 5: 完整推理循环与采样

- [x] 实现采样器（Greedy, Top-K, Top-P 等）
- [x] 实现推理引擎 `LLMEngine`，支持完整生成流程

### Day 6: 高级特性优化

- [x] 实现 Tensor Parallelism，支持多卡推理
- [x] 集成 CUDA Graph，优化 Decode 阶段性能

### Day 7: 测试与文档整理

- [x] 性能测试与调试
- [x] 整理文档与代码，准备简历描述

---

## 📊 性能数据

| 指标               | 本项目         | HuggingFace | 提升  |
|--------------------|----------------|-------------|-------|
| 吞吐量 (tokens/s) | 待测试         | 待测试      | 待测试 |
| 首 Token 延迟      | 待测试         | 待测试      | 待测试 |
| 显存占用           | 待测试         | 待测试      | 待测试 |

---

## 📚 参考资料

- [vLLM 论文](https://arxiv.org/abs/2309.06180)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)
- [nano-vllm 源码](https://github.com/GeeeekExplorer/nano-vllm)

---

## 📝 许可证

本项目遵循 MIT 许可证，详情请参阅 [LICENSE](./LICENSE) 文件。
```