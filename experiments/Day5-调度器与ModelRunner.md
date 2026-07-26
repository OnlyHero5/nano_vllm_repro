# Day 5 — 调度器与 ModelRunner

## 本篇定位

你已经有了：
- 数据结构（Sequence、Config、Context）
- 模型组件（RMSNorm、RoPE、融合 Linear）
- PagedAttention 引擎（BlockManager、Attention）

现在需要把它们串起来：**Scheduler 决定这轮算谁，ModelRunner 负责执行计算**。这两个模块是整个引擎的"大脑"和"双手"。

---

## 1. 知识点：Continuous Batching 调度器

### 1.1 为什么需要调度器？

一个推理引擎同时面对多个请求：

```
时刻 0: 请求A 到达 — "解释量子力学"（prompt 50 tokens）
时刻 1: 请求B 到达 — "写一首诗"（prompt 30 tokens）
时刻 2: 请求A 还在生成中...
时刻 3: 请求C 到达 — "1+1=?"（prompt 3 tokens）
```

GPU 显存是有限的（KV Cache 占用大户），每轮计算有 token budget 上限。

调度器的核心任务就是一个**经济学问题**：

> **在有限的 KV Cache 显存预算下，让 GPU 持续有活干。**

### 1.2 双队列设计

```
┌──────────────────────────────────────────────────────┐
│ Scheduler                                            │
│                                                      │
│ waiting (deque)              running (deque)         │
│ ┌──────┬──────┬──────┐      ┌──────┬──────┬──────┐  │
│ │ reqC │ reqD │ ...  │      │ reqA │ reqB │ ...  │  │
│ └──────┴──────┴──────┘      └──────┴──────┴──────┘  │
│  新来的，还没开始算              正在逐 token 生成     │
│  状态: WAITING                  状态: RUNNING        │
│  需要: Prefill（一次灌入全        需要: Decode（每次只算  │
│         部 prompt token）             1 个新 token）    │
└──────────────────────────────────────────────────────┘
```

每一步 `schedule()` 被调用时：

```
规则1（Prefill 优先）：
  从 waiting 队列头部取请求
  → 检查 KV Cache 够不够？
  → 检查 token budget 有没有超？
  → 够了就分配 blocks，移入 running 队列
  → 不够就停止（等下一轮）

规则2（Decode）：
  只有 waiting 空了才执行
  → 从 running 队列逐个取出
  → 检查是否需要新 block（每次 decode 最多需要 1 个新 slot）
  → 不够就抢占（preempt）最后加入的请求
```

### 1.3 为什么 Prefill 优先？

因为新请求的 prompt 可能很长（几百个 token），如果 GPU 等它慢慢生成，后面的短请求会被堵住。先让所有请求都"上车"（完成 prefill），再逐个推进 decode。

### 1.4 Preemption（抢占）

当显存不够时怎么办？抢占最后加入的 running 请求：

```python
while not block_manager.can_append(seq):
    if self.running:
        victim = self.running.pop()      # 抢占队尾的（LRU）
        self.__preempt(victim)           # 释放 KV Cache，放回 waiting
    else:
        self.__preempt(seq)              # 只剩自己了，只能抢占自己
        break
```

被抢占的请求下次会被重新 prefill（从头计算 KV Cache）。

---

## 2. 知识点：ModelRunner 的执行模型

### 2.1 Prefill vs Decode 的输入差异

这是理解 ModelRunner 最关键的区别：

| | Prefill 阶段 | Decode 阶段 |
|---|---|---|
| **输入** | 整段 prompt（可能 N 个 token） | 每个序列只有 1 个新 token |
| **QKV 形状** | `(N, hidden)` | `(batch_size, hidden)` |
| **Attention 模式** | `flash_attn_varlen_func`（一次性并行计算全部） | `flash_attn_with_kvcache`（只算新 Q 对历史 KV 的注意力） |
| **KV Cache** | 写入（整段 prompt 的 K/V） | 追加写入（1 个新 token 的 K/V） |
| **positions** | 从 0 开始递增 | 各自的当前位置 |

**Prefill 输入准备**：
```python
# 把所有序列的 token 拼成一个大序列
all_token_ids = [seq0_token0, seq0_token1, ..., seq1_token0, ...]
all_positions = [0, 1, 2, ..., 0, 1, ...]
cu_seqlens = [0, len(seq0), len(seq0)+len(seq1)]  # 累积长度，FlashAttention 需要
slot_mapping = [块号×块大小+偏移 for each token]
```

**Decode 输入准备**：
```python
# 每个序列只取最后一个 token
input_ids = [seq0.last_token, seq1.last_token]
positions = [seq0.num_tokens-1, seq1.num_tokens-1]
context_lens = [seq0.num_tokens, seq1.num_tokens]      # 历史的 KV 长度
block_tables = [[0, 2, 5, 0, 0], [1, 3, 0, 0, 0]]     # 填充到相同长度
slot_mapping = [最新 token 的 cache 位置 for each seq]
```

### 2.2 全局 Context 的生命周期

ModelRunner 每轮执行需要：
1. 设置 Context（`set_context()`）
2. 执行模型前向（Attention 层会去读 Context）
3. **清理** Context（`reset_context()`） ← 当前代码缺少这步！

如果忘记了 `reset_context()`，下一轮的 Context 可能混入上一轮的旧数据。这就是 **Context 泄漏**。

---

## 3. Scheduler 现在长什么样

### 3.1 代码结构

```python
class Scheduler:
    def __init__(self, config, block_manager):
        self.waiting: deque[Sequence] = deque()   # WAITING 状态
        self.running: deque[Sequence] = deque()   # RUNNING 状态
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = block_manager
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 阶段1: Prefill — 从 waiting 拉入
        # 阶段2: Decode — 从 running 逐个推进
        # 返回 (序列列表, 是否是prefill)
    
    def postprocess(self, seqs, token_ids):
        # 追加 token → 检查 EOS/max_tokens → 释放完成序列
    
    def __preempt(self, seq):
        # 抢占：释放 KV Cache，放回 waiting 队首
```

### 3.2 判断逻辑

```python
# Prefill: 检查是否还有 capacity
new_tokens = len(seq) - seq.num_cached_tokens
if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
    break  # token budget 不够

if not self.block_manager.can_allocate(seq):
    break  # KV Cache 不够
```

---

## 4. ModelRunner 现在长什么样

### 4.1 代码结构

```python
class ModelRunner:
    def __init__(self, config):
        # 加载 tokenizer、模型、分配 Sampler
    
    def allocate_kv_cache(self, num_blocks):
        # 预分配 KV Cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
    
    def prepare_prefill(self, sequences):
        # 拼接所有 token → 设置 Prefill Context
    
    def prepare_decode(self, sequences):
        # 取每个序列最后 token → 设置 Decode Context
    
    def run(self, sequences, is_prefill):
        # 准备输入 → 模型前向 → 采样 → 返回 token IDs
```

### 4.2 薄弱处

**问题1：`run()` 没有调用 `reset_context()`**

```python
# 当前代码：
def run(self, sequences, is_prefill):
    if is_prefill:
        input_ids, positions = self.prepare_prefill(sequences)
    else:
        input_ids, positions = self.prepare_decode(sequences)
    
    logits = self.model(input_ids, positions)
    # 取 logits...
    next_tokens = self.sampler(logits, temperatures)
    return next_tokens.tolist()
    # 问题：缺少 reset_context()！
```

**问题2：`LLMEngine.step()` 的 token 统计不准**

```python
# 当前代码：
if is_prefill:
    num_tokens = sum(len(seq) for seq in seqs)  # 问题：包含了已缓存的 prefix token！
```

已缓存的前缀 token 不应该被重复统计。

**问题3：Sampler 调用没有传 top_k/top_p**

等 Day6 完善 Sampler 后需要更新调用方式。

---

## 5. 需要修复的内容

| 修复项 | 文件 | 改动 |
|--------|------|------|
| `run()` 缺少 `reset_context()` | `engine/model_runner.py` | 用 try/finally 包裹，保证 Context 一定清理 |
| 拆出 `run_model()` | `engine/model_runner.py` | 为 Day7 CUDA Graph 做准备 |
| **`run_model()` 未同步 Day4 改动** | `engine/model_runner.py` | **forward() 返回 hidden_states，需调用 compute_logits()** |
| prefill token 统计不准 | `engine/llm_engine.py` | 扣除 `num_cached_tokens` |
| `postprocess()` 更新 cached 计数 | `engine/scheduler.py` | 在 append_token 前标记 num_cached_tokens |
| **Sampler 调用缺 top_k/top_p** | `engine/model_runner.py` | **从 Sequence 读取 top_k/top_p 并传给 Sampler** |

> **重要**：Day4 已把 `Qwen3ForCausalLM.forward()` 改成只返回 hidden_states，所以本篇的 `run_model()` 必须自己调 `compute_logits()`。
> 如果你跳过 Day4 直接看 Day5，先回去完成 forward/compute_logits 的拆分，否则这里拿不到 logits。

---

## 6. 完整代码

### 6.1 `engine/scheduler.py`（修复后）

```python
"""
Scheduler - Continuous Batching 调度器

核心职责：
1. 管理 waiting/running 两个队列（deque）
2. 决定每个 step 处理哪些序列（Prefill 优先）
3. 与 BlockManager 协作管理 KV Cache
4. 处理序列完成和 Preemption（抢占）

调度策略：
- Prefill 优先：新请求优先处理，让它尽快"上车"
- FCFS (First Come First Serve): 同一队列内先到先服务
- LRU Preemption: 内存不足时抢占最后加入 running 队列的请求

学习要点：
- 为什么 Prefill 优先？因为新请求可能很长，优先处理避免饥饿
- 为什么用 deque？因为 waiting 尾部插入 + 头部取出（FIFO），running 需要两端操作
"""

from collections import deque
from typing import Tuple, List

from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:
    """Continuous Batching 调度器

    维护两个队列：
    - waiting: 新请求（WAITING 状态），等待 Prefill
    - running: 正在生成中的请求（RUNNING 状态），进行 Decode
    """

    def __init__(self, config: Config, block_manager: BlockManager):
        # ── 从配置读取限制参数 ──
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos  # 结束符 token ID

        # ── 与显存管理的桥梁 ──
        self.block_manager = block_manager

        # ── 双队列（deque: 双端队列，两端操作都是 O(1)）──
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    # ─── 状态查询 ───

    def is_finished(self) -> bool:
        """所有请求是否都已完成"""
        return len(self.waiting) == 0 and len(self.running) == 0

    def get_num_waiting(self) -> int:
        """等待队列长度"""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """运行队列长度"""
        return len(self.running)

    # ─── 添加请求 ───

    def add_sequence(self, seq: Sequence):
        """将新序列加入 waiting 队列

        调用时机：LLMEngine.add_request() 时
        """
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    add = add_sequence  # 别名，两种写法都可以

    # ─── 核心调度逻辑 ───

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """核心调度：决定本轮处理哪些序列

        返回值：
            (scheduled_seqs, is_prefill)
            - scheduled_seqs: 本轮要处理的序列列表
            - is_prefill: True=Prefill阶段, False=Decode阶段

        调度逻辑：
            阶段1（Prefill）：从 waiting 队列取请求，检查显存和 token budget
            阶段2（Decode）：只有 waiting 空了才执行，从 running 逐个推进
        """
        scheduled_seqs: List[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0

        # ═══════════════════════════════════════
        # 阶段1：Prefill — 让新请求"上车"
        # ═══════════════════════════════════════
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 看队首但不取出（可能不满足条件）

            # 检查1：token budget 够不够？
            # num_cached_tokens 是 prefix cache 命中的部分，不需要重新计算
            new_tokens = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
                break  # 这轮装不下了，等下一轮

            # 检查2：KV Cache 显存够不够？
            if not self.block_manager.can_allocate(seq):
                break  # 显存不够，等下一轮（可能有请求完成释放显存）

            # 通过检查 → 分配 KV Cache，移入 running 队列
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()       # 从 waiting 移除
            self.running.append(seq)     # 加入 running
            scheduled_seqs.append(seq)

            num_seqs += 1
            num_batched_tokens += new_tokens

        # 如果有 prefill 任务，立即返回（Prefill 优先）
        if scheduled_seqs:
            return scheduled_seqs, True

        # ═══════════════════════════════════════
        # 阶段2：Decode — 推进已在生成中的请求
        # ═══════════════════════════════════════
        decoded_seqs: List[Sequence] = []

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 从队首取出

            # 检查是否需要新 slot（decode 每次只需要追加 1 个 slot）
            while not self.block_manager.can_append(seq):
                # 显存不够 → 需要抢占（preempt）
                if self.running:
                    # 抢占队尾的请求（LRU: 最后加入的最先被抢占）
                    victim = self.running.pop()
                    self.__preempt(victim)
                else:
                    # 只剩当前请求了，只能抢占自己（防止死锁）
                    self.__preempt(seq)
                    break
            else:
                # can_append 通过 → 分配 slot
                self.block_manager.append_slot(seq)
                decoded_seqs.append(seq)
                num_seqs += 1

        # 把处理过的序列放回 running 队首（保持顺序）
        for seq in reversed(decoded_seqs):
            self.running.appendleft(seq)

        return decoded_seqs, False

    def __preempt(self, seq: Sequence):
        """抢占：释放序列的 KV Cache，放回 waiting 队列队首

        Preemption（抢占）是保证系统在显存紧张时不会死锁的关键机制。
        被抢占的请求下次会被重新 Prefill（从头计算 KV Cache）。

        为什么放队首而不是队尾？
        → 这个请求已经等了很久，应该尽快重新处理，避免饥饿。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)  # 放队首，优先重新计算

    # ─── 后处理 ───

    def postprocess(
        self,
        seqs: List[Sequence],
        token_ids: List[int],
    ) -> List[Sequence]:
        """后处理：模型推理完成后的状态更新

        在模型 forward 和采样之后调用，做三件事：
        1. 标记本轮 token 已写入 KV Cache
        2. 将生成的新 token 追加到序列
        3. 检查终止条件（EOS 或达到 max_tokens），释放已完成序列

        Args:
            seqs: 本轮处理的序列列表
            token_ids: 每个序列新生成的 token ID

        Returns:
            已完成的序列列表
        """
        finished_seqs: List[Sequence] = []

        for seq, token_id in zip(seqs, token_ids):
            # ── 步骤1：标记本轮 token 已进入 KV Cache ──
            # 为什么要在这步做？因为本轮 ModelRunner 已经把 K/V 写入了 cache
            seq.num_cached_tokens = max(seq.num_cached_tokens, seq.num_tokens)

            # ── 步骤2：追加新 token ──
            seq.append_token(token_id)

            # ── 步骤3：检查终止条件 ──
            is_eos = (not seq.ignore_eos) and (token_id == self.eos)
            is_max_tokens = seq.num_completion_tokens >= seq.max_tokens

            if is_eos or is_max_tokens:
                # 序列完成 → 释放资源
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放 KV Cache 页
                self.running.remove(seq)            # 移出 running 队列
                finished_seqs.append(seq)

        return finished_seqs

    def __repr__(self) -> str:
        return (
            f"Scheduler(waiting={self.get_num_waiting()}, "
            f"running={self.get_num_running()}, "
            f"free_blocks={self.block_manager.get_num_free_blocks()})"
        )
```

### 6.2 `engine/model_runner.py`（修复后）

```python
"""
ModelRunner - 模型执行器

职责：
1. 加载模型和 tokenizer
2. 分配 KV Cache 显存
3. 准备模型输入（构建 Context）
4. 执行模型前向传播
5. 调用 Sampler 生成 token
6. 管理 Context 生命周期

这是连接 Scheduler 和 Model 的桥梁，也是整个推理流程中
"把调度决策转化为 GPU 计算"的关键环节。
"""

import torch
from torch import nn
from transformers import AutoTokenizer
from typing import Optional

from config import Config
from engine.sequence import Sequence
from utils.context import Context, set_context, get_context, reset_context
from utils.loader import load_model
from layers.sampler import Sampler


class ModelRunner:
    """模型执行器

    核心流程：
    ┌──────────────────────────────────────────────────────┐
    │ run(sequences, is_prefill)                          │
    │   ├─ prepare_prefill() / prepare_decode()           │
    │   │   └─ 构建 input_ids, positions, Context         │
    │   ├─ run_model()                                    │
    │   │   ├─ model.forward()  → hidden_states           │
    │   │   └─ model.compute_logits() → logits            │
    │   ├─ sampler() → next_tokens                       │
    │   └─ reset_context()  ← finally 块保证一定执行       │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载 tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        # ── 加载模型 ──
        self.model = self._load_model()

        # ── 采样器（Day6 会升级为支持 top_k/top_p）──
        self.sampler = Sampler()

        # ── KV Cache（在 allocate_kv_cache() 中分配）──
        self.kv_cache: Optional[list[torch.Tensor]] = None

        # ── 缓存模型配置字段（避免反复查 config）──
        self.num_layers = self.model.config.num_hidden_layers
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = getattr(
            self.model.config, "head_dim",
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )
        self.block_size = Sequence.block_size

    def _load_model(self) -> nn.Module:
        """加载 Qwen3 模型并迁移到 GPU"""
        from models.qwen3 import Qwen3ForCausalLM

        print(f"[ModelRunner] 加载模型：{self.config.model_path}")

        # 步骤1: 创建模型结构（随机初始化权重）
        model = Qwen3ForCausalLM.from_pretrained(self.config.model_path)

        # 步骤2: 从 safetensors 加载真实权重
        load_model(model, self.config.model_path)

        # 步骤3: 迁移到 GPU + bfloat16 + 评估模式
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()

        print(f"[ModelRunner] 模型加载完成，设备：{self.device}")
        return model

    # ─── KV Cache 显存分配 ───

    def allocate_kv_cache(self, num_blocks: int):
        """预分配 KV Cache 显存

        KV Cache 数据结构（每层一个 tensor）：
            shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
            - dim 0: K 和 V（索引 0 是 K，索引 1 是 V）
            - dim 1: 块索引（每个块存 block_size 个 token 的 KV）
            - dim 2: 块内 token 位置（0 ~ block_size-1）
            - dim 3: KV 头数（GQA 场景下小于 Q 头数）
            - dim 4: 每个头的维度

        显存计算（fp16，每个元素 2 bytes）：
            每层 = 2 × num_blocks × block_size × num_kv_heads × head_dim × 2 bytes
            总计 = num_layers × 每层

        为什么用 fp16 存 KV Cache？
        - KV Cache 对精度不敏感（只是查表）
        - bf16 也能用，但 fp16 体积更小、读写更快
        """
        bytes_per_block = (
            2 *                    # K 和 V
            self.block_size *      # 每块的 token 数
            self.num_kv_heads *    # KV 头数
            self.head_dim *        # 每头维度
            2                       # fp16 = 2 bytes
        )
        total_bytes = self.num_layers * num_blocks * bytes_per_block
        print(f"[ModelRunner] KV Cache 显存需求：{total_bytes / 1024**3:.2f} GB")

        self.kv_cache = []
        for _ in range(self.num_layers):
            cache = torch.zeros(
                2, num_blocks, self.block_size,
                self.num_kv_heads, self.head_dim,
                dtype=torch.float16,
                device=self.device,
            )
            self.kv_cache.append(cache)

        print(f"[ModelRunner] KV Cache 分配完成：{num_blocks} 块 × {self.num_layers} 层")

    def get_num_free_gpu_blocks(self) -> int:
        """根据 GPU 空闲显存计算可分配的 KV Cache 块数

        考虑了 gpu_memory_utilization 系数（默认 0.7），
        即只使用空闲显存的 70% 给 KV Cache。
        """
        if not torch.cuda.is_available():
            return 100  # CPU 模式下的默认值

        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory

        # 只使用一部分空闲显存（留给模型权重和其他开销）
        available_memory = free_memory * self.config.gpu_memory_utilization

        # 每块的显存需求
        bytes_per_block_per_layer = (
            2 * self.block_size * self.num_kv_heads * self.head_dim * 2
        )
        bytes_per_block = bytes_per_block_per_layer * self.num_layers

        num_blocks = int(available_memory // bytes_per_block)

        print(f"[ModelRunner] GPU显存：{total_memory / 1024**3:.1f} GB 总计, "
              f"{free_memory / 1024**3:.1f} GB 空闲")
        print(f"[ModelRunner] 可分配 KV Cache 块数：{num_blocks}")

        return num_blocks

    # ─── 输入准备 ───

    def prepare_prefill(
        self,
        sequences: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """准备 Prefill 阶段的输入

        Prefill 特点：
        - 一次性把整段 prompt 的全部 token 送入模型
        - FlashAttention 用 varlen API（cu_seqlens 标记各序列边界）
        - 每个 token 的 K/V 写入 KV Cache 指定的 slot

        返回：
            (input_ids, positions)
            同时设置全局 Context 供 Attention 层读取。
        """
        all_token_ids: list[int] = []
        all_positions: list[int] = []
        cu_seqlens = [0]       # 累积序列长度：[0, len_s0, len_s0+len_s1, ...]
        slot_mapping: list[int] = []  # 每个 token 在 KV Cache 中的精确槽位

        for seq in sequences:
            token_ids = seq.token_ids
            seq_len = len(token_ids)

            # ── 拼接 token IDs 和 positions ──
            all_token_ids.extend(token_ids)
            all_positions.extend(range(seq_len))

            # ── 记录累积长度（FlashAttention varlen API 需要）──
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

            # ── 计算每个 token 的 slot ──
            # slot = 物理页号 × block_size + 页内偏移
            for i in range(seq_len):
                block_idx = i // self.block_size      # 第几个逻辑块
                offset = i % self.block_size           # 块内偏移
                if block_idx < len(seq.block_table):
                    block_id = seq.block_table[block_idx]  # 查表：逻辑块→物理页
                    slot = block_id * self.block_size + offset
                    slot_mapping.append(slot)
                else:
                    slot_mapping.append(0)  # 防御性：不应该走到这里

        # ── 转为 GPU 张量 ──
        input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
        max_seqlen = max(len(seq.token_ids) for seq in sequences)

        # ── 设置全局 Context ──
        context = Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            # Prefill 阶段不需要 context_lens 和 block_tables
            context_lens=None,
            block_tables=None,
            max_context_len=None,
            max_num_blocks=None,
            kv_cache=self.kv_cache,
        )
        set_context(context)
        return input_ids, positions

    def prepare_decode(
        self,
        sequences: list[Sequence],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """准备 Decode 阶段的输入

        Decode 特点：
        - 每个序列只取最新生成的 1 个 token（last_token）
        - FlashAttention 用 with_kvcache API（只算新 Q 对历史的注意力）
        - 需要传 block_tables 让 FlashAttention 知道去哪里读历史 KV

        返回：
            (input_ids, positions)
        """
        input_ids: list[int] = []
        positions: list[int] = []
        context_lens: list[int] = []       # 每个序列的上下文已有多长
        block_tables: list[list[int]] = []  # 每个序列的 block_table
        slot_mapping: list[int] = []        # 新 token 写入 KV Cache 的位置

        # 计算最大块数（所有序列 block_table 的 max 长度，用于 padding）
        max_num_blocks = max(
            (len(seq.block_table) for seq in sequences), default=0
        )

        for seq in sequences:
            # ── 只取最后一个 token ──
            input_ids.append(seq.last_token)

            # ── 位置从 0 开始，即当前总长度 - 1 ──
            positions.append(seq.num_tokens - 1)

            # ── 上下文长度 ──
            context_lens.append(seq.num_tokens)

            # ── block_table（填充到相同长度，FlashAttention 要求）──
            padded = seq.block_table.copy()
            while len(padded) < max_num_blocks:
                padded.append(0)  # 用 0 填充
            block_tables.append(padded)

            # ── 新 token 的 slot ──
            pos = seq.num_tokens - 1  # 这是第几个 token（0-indexed）
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = seq.block_table[block_idx] if block_idx < len(seq.block_table) else 0
            slot = block_id * self.block_size + offset
            slot_mapping.append(slot)

        # ── 转为 GPU 张量 ──
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)

        # ── 设置全局 Context（Decode 版本）──
        context = Context(
            is_prefill=False,
            cu_seqlens_q=None,    # Decode 不需要累积长度
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),
            max_context_len=max(context_lens) if context_lens else 0,
            max_num_blocks=max_num_blocks,
            kv_cache=self.kv_cache,
        )
        set_context(context)
        return input_ids, positions

    # ─── 模型执行（拆出独立方法，为 CUDA Graph 做准备）───

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        """执行模型主干计算，返回 Sampler 需要的 logits

        拆出 run_model() 而不是在 run() 里一把梭的原因：
        - Day7 实现 CUDA Graph 时只需要替换这个方法
        - 方便单独 benchmark 模型前向的性能
        - 职责清晰：run_model = 纯计算，run = 流程编排
        """
        # ── 步骤1：模型前向 ──
        # Day4 改造后 forward() 只返回 hidden_states（不经过 lm_head）
        # 这样 CUDA Graph 录制时不需要捕获巨大的 vocab 投影
        hidden_states = self.model(input_ids, positions)

        # ── 步骤2：Prefill 时取每个序列最后一个位置的 hidden states ──
        # 为什么只要最后一个？因为采样只需要最后一个位置的预测来决定下一个 token
        if is_prefill:
            context = get_context()
            # cu_seqlens_q 是 [0, len_seq0, len_seq0+len_seq1, ...]
            # [1:] - 1 得到每个序列的最后一个位置索引
            last_token_indices = context.cu_seqlens_q[1:] - 1
            last_token_indices = last_token_indices.long()
            hidden_states = hidden_states[last_token_indices]
        # Decode 阶段：每条序列已经只有 1 个 token，hidden_states 形状是 (batch, hidden_size)

        # ── 步骤3：vocab 投影 ──
        # compute_logits() 将 hidden_states 投影到 vocab 维度
        logits = self.model.compute_logits(hidden_states)

        return logits

    # ─── 主入口 ───

    @torch.inference_mode()
    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:
        """执行当前 step 的完整推理流程

        Args:
            sequences: 本轮要处理的序列列表
            is_prefill: True=Prefill, False=Decode

        Returns:
            next_tokens: 每个序列的下一个 token ID 列表
        """
        if not sequences:
            return []

        # ── 步骤1：准备输入（同时设置 Context）──
        if is_prefill:
            input_ids, positions = self.prepare_prefill(sequences)
        else:
            input_ids, positions = self.prepare_decode(sequences)

        try:
            # ── 步骤2：模型前向 ──
            logits = self.run_model(input_ids, positions, is_prefill)

            # ── 步骤3：采样（从 logits 选出下一个 token）──
            # 传递 temperature、top_k、top_p 给 Sampler
            # Day6 会实现这些采样策略的具体逻辑
            temperatures = torch.tensor(
                [seq.temperature for seq in sequences],
                dtype=torch.float32,
                device=self.device,
            )
            top_ks = torch.tensor(
                [seq.top_k for seq in sequences],
                dtype=torch.int32,
                device=self.device,
            )
            top_ps = torch.tensor(
                [seq.top_p for seq in sequences],
                dtype=torch.float32,
                device=self.device,
            )
            next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
            return next_tokens.tolist()

        finally:
            # ── 步骤4：无论如何都要清理 Context ──
            # finally 块保证：即使模型报错，Context 也会被清理
            # 这防止了 Context 泄漏到下一轮 step
            reset_context()
```

### 6.3 `engine/llm_engine.py`（修复 token 统计）

```python
"""
LLM 推理引擎

串联 Scheduler + ModelRunner，实现完整推理循环。

职责：
1. 初始化各组件（Tokenizer、ModelRunner、BlockManager、Scheduler）
2. 对外提供 add_request() 和 generate() 接口
3. step() 驱动一轮「调度→推理→后处理」
4. 用 tqdm 显示吞吐量进度条
"""

import atexit
from time import perf_counter
from typing import Union

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from config import Config
from sampling_params import SamplingParams
from engine.sequence import Sequence
from engine.block_manager import BlockManager
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner


class LLMEngine:
    """LLM 推理引擎

    这是整个项目的顶层入口。用户通过 LLM.generate() 调用，
    内部由 LLMEngine 协调所有组件完成推理。
    """

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: 模型路径（如 "models/Qwen3-0.6B"）
            **kwargs: 可选配置参数（如 enforce_eager=True）
        """
        # ── 创建配置（Config.__post_init__ 会验证模型路径并加载 HF 配置）──
        self.config = Config(model_path=model, **kwargs)

        # ── 加载 Tokenizer ──
        print(f"[LLMEngine] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self.config.eos = self.tokenizer.eos_token_id
        print(f"[LLMEngine] EOS token ID: {self.config.eos}")

        # ── 初始化 ModelRunner（加载模型 + 采样器）──
        print(f"[LLMEngine] 初始化 ModelRunner...")
        self.model_runner = ModelRunner(self.config)

        # ── 计算并分配 KV Cache 块 ──
        num_blocks = self.model_runner.get_num_free_gpu_blocks()
        num_blocks = max(1, int(num_blocks * 0.95))  # 留 5% 余量
        self.model_runner.allocate_kv_cache(num_blocks)

        # ── 创建 BlockManager（物理块池管理）──
        block_size = Sequence.block_size
        self.block_manager = BlockManager(num_blocks, block_size)

        # ── 创建 Scheduler（调度器）──
        self.scheduler = Scheduler(self.config, self.block_manager)

        # ── 注册退出时的清理逻辑 ──
        atexit.register(self._cleanup)

        print(f"[LLMEngine] 初始化完成！")
        print(f"[LLMEngine] - KV Cache: {num_blocks} 块")
        print(f"[LLMEngine] - Block Size: {block_size} tokens")

    def _cleanup(self):
        """清理 GPU 显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── 添加请求 ───

    def add_request(
        self,
        prompt: Union[str, list[int]],
        sampling_params: SamplingParams = None,
    ):
        """添加一个推理请求

        这是用户请求的入口。做了两件事：
        1. Tokenize（字符串 → token ID 列表）
        2. 创建 Sequence 并加入 Scheduler 的 waiting 队列

        Args:
            prompt: 文本输入（str）或已 tokenize 的 token ID 列表
            sampling_params: 采样配置（temperature、max_tokens 等）
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Tokenize：将文本转为 token ID 列表
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)

        # 封装为 Sequence 并加入调度器
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add_sequence(seq)

    # ─── 状态查询 ───

    def is_finished(self) -> bool:
        """所有请求是否都已完成"""
        return self.scheduler.is_finished()

    # ─── 单步推理 ───

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """执行单步推理：调度 → 模型前向 → 采样 → 后处理

        返回值：
            (outputs, num_tokens)
            - outputs: [(seq_id, [completion_token_ids]), ...] 完成的序列
            - num_tokens: 本轮处理的 token 数
                * 正数 = prefill 阶段新计算的 token 数（不含已缓存前缀）
                * 负数 = decode 阶段，绝对值 = 处理的序列数

        为什么 num_tokens 用符号区分？
        → 上层 generate() 做吞吐统计时，prefill 和 decode 的"工作量"含义不同。
          prefill 是「算了多少新 token」，decode 是「推进了多少个序列」。
        """
        # ── 步骤1：调度（决定本轮算谁）──
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0

        # ── 步骤2：模型推理 ──
        token_ids = self.model_runner.run(seqs, is_prefill)

        # ── 步骤3：后处理（更新状态、释放完成序列）──
        finished_seqs = self.scheduler.postprocess(seqs, token_ids)

        # ── 步骤4：收集输出 ──
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in finished_seqs
        ]

        # ── 步骤5：计算 token 统计 ──
        # 修复：prefill 只统计本轮新计算的 token（不含已缓存前缀）
        if is_prefill:
            num_tokens = sum(
                len(seq) - seq.num_cached_tokens
                for seq in seqs
            )
            # 注意：这里 num_cached_tokens 是 prefill 之前的缓存数
            # postprocess 在 step() 里会在 model_runner.run() 之后更新它
            # 但我们在这里取的是 scheduler.schedule() 之后的值
            # 此时 postprocess 还没执行，所以 num_cached_tokens 是准确的"之前缓存"
        else:
            num_tokens = -len(seqs)  # decode: 负数表示序列数

        return outputs, num_tokens

    # ─── 批量生成（顶层 API）───

    def generate(
        self,
        prompts: Union[list[str], list[list[int]]],
        sampling_params: Union[SamplingParams, list[SamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """批量生成文本

        这是对外暴露的主接口。典型用法：
            llm = LLM("models/Qwen3-0.6B")
            outputs = llm.generate(["你好", "1+1=?"])
            print(outputs[0]["text"])

        Args:
            prompts: 输入文本列表（str）或已 tokenize 的 token ID 列表
            sampling_params: 统一的或逐条定制的采样参数
            use_tqdm: 是否显示 tqdm 进度条

        Returns:
            results: [{"text": str, "token_ids": [int]}, ...]
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # 统一为列表形式
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # ── 添加所有请求到调度器 ──
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # ── tqdm 进度条 + 吞吐量监控 ──
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}  # seq_id → completion_token_ids
        prefill_throughput = 0.0
        decode_throughput = 0.0

        # ── 核心生成循环 ──
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - t

            # 更新进度条
            if pbar and elapsed > 0:
                if num_tokens > 0:  # Prefill
                    prefill_throughput = num_tokens / elapsed
                elif num_tokens < 0:  # Decode
                    decode_throughput = -num_tokens / elapsed
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })

            # 收集完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        # ── 按 seq_id 排序，保证输出顺序与输入一致 ──
        sorted_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # ── Tokenize 回文本 ──
        results = []
        for token_ids in sorted_outputs:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": token_ids})

        return results
```

---

## 7. 验证步骤

```bash
cd nano_vll_repro

# 1. 语法检查
python -m py_compile engine/scheduler.py engine/model_runner.py engine/llm_engine.py

# 2. 跑已有测试确认没破坏
python tests/test_Day3.py
```

> **注意**：如果 `test_Day3.py` 报错，请参考 Day3 指南「验证步骤」中的 bug 修复说明。

```bash
# 3. 快速手动验证 Context 清理
python - <<'PY'
from utils.context import get_context, reset_context
ctx = get_context()
print(f"初始 is_prefill: {ctx.is_prefill}")
reset_context()
ctx = get_context()
print(f"重置后 max_num_blocks: {ctx.max_num_blocks}")  # 应为 None
print("Context 生命周期验证通过")
PY

# 4. 端到端推理（需要有模型权重）
python example.py
```

预期输出：
- Day3 测试通过
- Context 验证脚本输出 `max_num_blocks: None`
- `example.py` 正常生成文本，进度条显示 Prefill 和 Decode 吞吐量

---

## 8. 本篇总结

| 模块 | 核心职责 | 关键设计 |
|------|---------|---------|
| **Scheduler** | 决定每步算谁 | 双队列 + Prefill 优先 + LRU 抢占 |
| **ModelRunner** | 执行计算 | Prefill/Decode 两套输入准备 + Context 管理 |
| **LLMEngine** | 顶层循环 | step() 串起一切 + generate() 批量接口 |

**最重要的三件事**：
1. **Prefill 优先**确保新请求不被长请求饿死
2. **slot_mapping** 是 Attention 写入 KV Cache 的关键——每个 token 写在哪里
3. **try/finally reset_context()** 防止 Context 泄漏到下一轮

下一篇：**Day6 — 完整推理链路与 Sampler 完善**（Sampler 升级、example.py 优化）
