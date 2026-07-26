# Day 1 — 数据结构层：Config / SamplingParams / Sequence / Context

## 本篇定位

这 4 个文件是整个项目的「骨架」。它们不涉及 GPU 计算，不涉及模型，只是**定义数据如何组织、配置如何管理、状态如何流转**。

这一层的代码已经能跑，但留着几个坑。本篇一边把每个数据结构读透，一边**把这些坑逐个填上**。

---

## 1. 知识点：为什么这四个类要这样设计？

### 1.1 Config — 一个数据中心，不是配置文件的搬运工

HuggingFace 的模型配置（`config.json`）里的字段命名不统一：
- 有的叫 `hidden_size`，有的叫 `hidden_dim`
- 有的叫 `num_attention_heads`，有的叫 `n_head`
- RoPE 参数可能叫 `rope_theta`，也可能藏在 `rope_parameters` 字典里

如果让下游代码到处写 `getattr(hf_config, 'xxx', default)`，会出现两个问题：
1. **代码散落**：同一个 fallback 逻辑出现在 5 个文件里
2. **排查困难**：如果某个属性获取方式不对，你不知道该改哪里

**解决方案**：Config 类提供统一的 `@property`，把 HF 的混乱字段翻译成稳定名字。下游使用者只需 `config.hidden_size`，不需要关心 HF 里到底叫啥。

### 1.2 SamplingParams — 让每条请求有自己的生成风格

不同任务需要不同的采样策略：

| 任务 | temperature | top_k | top_p |
|------|------------|-------|-------|
| 代码生成 | 0（greedy） | — | — |
| 创意写作 | 0.8 | 20 | 0.9 |
| 翻译 | 0.3 | 40 | 1.0 |

所以采样参数**不是全局的**——它们属于每条请求。进入系统后，SamplingParams 会复制到 Sequence 对象里。

### 1.3 Sequence — 一条请求的「身份证」

Sequence 封装了一条请求从生到死的全部运行时状态：

```
创建时：status=WAITING, token_ids=[prompt], block_table=[]
调度后：status=RUNNING, block_table=[物理页17, 物理页203, ...]
结束时：status=FINISHED, 释放 block_table
```

最关键的属性是 **`block_table`**——它记录了「这条请求的 KV Cache 存在哪些物理页里」。这就是 PagedAttention 的核心数据结构（类比操作系统的页表）。

### 1.4 Context — 跨层数据传递的「全局信箱」

模型前向传播经过 Embedding → DecoderLayer × 28 → Norm → LM Head。只有 **Attention 层**需要知道本轮推理的元数据（slot_mapping、block_tables 等）。

如果把元数据通过函数参数层层传递，需要改 28 个 DecoderLayer 的 forward 签名。Context 用全局单例绕过了这一问题——就像一个全局信箱，ModelRunner 往里放，Attention 从中取。

```
ModelRunner.prepare_prefill()  ──→  set_context(ctx)  ──→  全局单例
                                                              │
Attention.forward()  ←──  ctx = get_context()  ←──────────────┘
```

---

## 2. 这一层现在长什么样

### 2.1 `config.py` — 当前状态

```python
# 当前已有的关键字段：
model_path: str                      # 模型路径
max_num_batched_tokens: int = 16384  # 单批次最大 token 数
max_num_seqs: int = 512              # 最大并发序列数
max_model_len: int = 4096            # 最大上下文长度
gpu_memory_utilization: float = 0.7  # 显存利用率
tensor_parallel_size: int = 1        # 张量并行数
enforce_eager: bool = False          # True=禁用 CUDA Graph
kvcache_block_size: int = 256        # KV Cache 块大小
num_kvcache_blocks: int = -1         # 块数量（运行时计算）
hf_config: AutoConfig | None = None  # HuggingFace 原始配置
eos: int = -1                        # EOS token ID
```

`__post_init__` 里做了：
- 校验模型路径存在
- 校验 block_size 是 256 的倍数
- 自动加载 `AutoConfig`
- 校验收敛 token 数够不够

### 2.2 `sampling_params.py` — 当前状态

```python
temperature: float = 1.0    # 温度（当前校验 temperature > 1e-10——这有 bug）
max_tokens: int = 4096      # 最多生成 token 数
ignore_eos: bool = False    # 是否忽略 EOS
```

### 2.3 `engine/sequence.py` — 当前状态

已实现的功能很完整：
- `SequenceStatus` 枚举（WAITING / RUNNING / FINISHED）
- token 管理（`token_ids`, `append_token`, `num_completion_tokens`）
- block 计算（`num_blocks`, `block(i)`, `last_block_num_tokens`）
- 序列化支持（`__getstate__` / `__setstate__` 用于多进程通信）
- prefix cache 账本（`num_cached_tokens`）

### 2.4 `utils/context.py` — 当前状态

已实现：
- `Context` dataclass（含 prefill/decode 两套字段）
- `set_context()` / `get_context()` / `reset_context()`

---

## 3. 这一版的薄弱处

### 问题 1：Config 缺少 property

当前 `model_runner.py` 里这样写：
```python
self.num_kv_heads = self.model.config.num_key_value_heads
self.head_dim = getattr(self.model.config, "head_dim", ...)
```

这是 **走后门**——明明有 Config 对象，却绕过它直接访问 HF config。应该把翻译逻辑收到 Config 的 property 里。

### 问题 2：SamplingParams 不能设置 temperature=0

当前校验是：
```python
assert self.temperature > 1e-10, "temperature 必须 > 0"
```

但 **temperature=0 是 greedy 解码**，这是非常常用的设置（代码生成、确定性输出等）。当前必须设一个极小的值（如 0.0001）来模拟 greedy，这不够干净。

### 问题 3：SamplingParams 缺 top_k / top_p

当前只有 temperature 和 max_tokens。需要补齐完整采样参数。

### 问题 4：Sequence 没有复制 top_k / top_p

Sequence 目前只复制了 `temperature`、`max_tokens`、`ignore_eos`。需要同步增加。

### 问题 5：`utils/context.py` 类型注解错误

当前代码：
```python
max_context_len: int = None   # None 不是 int
max_num_blocks: int = None    # None 不是 int
```

`int = None` 在类型检查器（mypy / pyright）中会报错，因为 `None` 不是 `int` 的合法值。应改为：
```python
max_context_len: int | None = None
max_num_blocks: int | None = None
```

这是一个**静态类型问题**，运行时不会报错（Python 不强制 dataclass 类型），但会影响 IDE 自动补全和类型检查。

---

## 4. 完善后的完整代码

### 4.1 `config.py`（完善版）

```python
"""
nano-vLLM 全局配置

Config 是本项目的"数据中心"——它把 HuggingFace 模型配置翻译成统一的名字，
同时管理推理引擎的运行时参数。

设计原则：
- 所有从 HF config 读取的字段，都通过 @property 暴露
- 调用方不需要知道 HF 原始字段叫什么名字
- fallback 逻辑集中在这里，不散落到其他文件
"""

import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass
class Config:
    """
    nano-vLLM 核心配置类

    两种字段：
    1. 用户可配置的（有默认值）——如 max_num_batched_tokens
    2. 运行时自动填充的 ——如 hf_config, eos
    """

    # ═══════════════════════════════════════
    # 模型路径
    # ═══════════════════════════════════════
    model_path: str

    # ═══════════════════════════════════════
    # 连续批处理参数
    # ═══════════════════════════════════════
    max_num_batched_tokens: int = 16384  # 单批次最大 token 数（含 prefill + decode）
    max_num_seqs: int = 512              # 最大并发序列数
    max_model_len: int = 4096            # 最大上下文长度

    # ═══════════════════════════════════════
    # 显存管理
    # ═══════════════════════════════════════
    gpu_memory_utilization: float = 0.7  # 显存利用率（0~1）

    # ═══════════════════════════════════════
    # 并行配置
    # ═══════════════════════════════════════
    tensor_parallel_size: int = 1        # 张量并行数（1 为单卡）

    # ═══════════════════════════════════════
    # 调试选项
    # ═══════════════════════════════════════
    enforce_eager: bool = False          # True = 禁用 CUDA Graph

    # ═══════════════════════════════════════
    # 运行时自动填充（用户不应手动设置）
    # ═══════════════════════════════════════
    hf_config: AutoConfig | None = None  # HuggingFace 原始模型配置
    eos: int = -1                        # 结束符 token ID

    # ═══════════════════════════════════════
    # PagedAttention 参数
    # ═══════════════════════════════════════
    kvcache_block_size: int = 256        # KV Cache 块大小（必须是 256 的倍数）
    num_kvcache_blocks: int = -1         # KV Cache 块数量（运行时根据显存计算）

    # ═══════════════════════════════════════
    # 初始化后校验
    # ═══════════════════════════════════════
    def __post_init__(self):
        """dataclass 初始化后自动调用，用于参数校验和自动配置"""

        # 1. 校验模型路径存在
        assert os.path.isdir(self.model_path), f"模型路径不存在：{self.model_path}"

        # 2. 块大小必须是 256 的倍数（FlashAttention 的 alignment 要求）
        assert self.kvcache_block_size % 256 == 0, (
            f"kvcache_block_size 必须是 256 的倍数，当前值：{self.kvcache_block_size}"
        )

        # 3. 张量并行数范围检查（单机 8 卡以内）
        assert 1 <= self.tensor_parallel_size <= 8, (
            f"张量并行数必须在 1-8 之间，当前值：{self.tensor_parallel_size}"
        )

        # 4. 自动加载 HuggingFace 模型配置
        self.hf_config = AutoConfig.from_pretrained(self.model_path)

        # 5. 上下文长度取配置文件和用户指定的最小值
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )

        # 6. 单批次 token 数必须 >= 最大上下文长度（确保能处理最长序列）
        assert self.max_num_batched_tokens >= self.max_model_len, (
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) "
            f"必须 >= max_model_len ({self.max_model_len})"
        )

    # ═══════════════════════════════════════
    # Property：统一翻译 HF 配置字段
    # ═══════════════════════════════════════

    @property
    def model(self) -> str:
        """别名：返回模型路径"""
        return self.model_path

    @property
    def hidden_size(self) -> int:
        """模型隐藏层维度。Qwen3-0.6B: 1024"""
        return self.hf_config.hidden_size

    @property
    def num_attention_heads(self) -> int:
        """Query 头数。Qwen3-0.6B: 16"""
        return self.hf_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        """KV 头数。GQA 时小于 num_attention_heads。Qwen3-0.6B: 8"""
        return getattr(
            self.hf_config,
            "num_key_value_heads",
            self.num_attention_heads,  # 如果模型不用 GQA，KV 头数等于 Q 头数
        )

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。Qwen3 显式提供，否则用 hidden_size // num_heads"""
        return getattr(
            self.hf_config,
            "head_dim",
            self.hidden_size // self.num_attention_heads,
        )

    @property
    def hidden_act(self) -> str:
        """激活函数类型。Qwen3: silu"""
        return getattr(self.hf_config, "hidden_act", "silu")

    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return self.hf_config.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        """Transformer 层数。Qwen3-0.6B: 28"""
        return self.hf_config.num_hidden_layers

    @property
    def intermediate_size(self) -> int:
        """FFN 中间层维度"""
        return self.hf_config.intermediate_size

    @property
    def rms_norm_eps(self) -> float:
        """RMSNorm 的 epsilon"""
        return getattr(self.hf_config, "rms_norm_eps", 1e-6)

    @property
    def max_position_embeddings(self) -> int:
        """模型支持的最大位置"""
        return self.hf_config.max_position_embeddings

    # ═══════════════════════════════════════
    # RoPE 参数
    # ═══════════════════════════════════════

    @property
    def rope_theta(self) -> float:
        """
        RoPE 的 base 频率。

        读取优先级：
        1. rope_parameters 字典里的 rope_theta 或 base
        2. 顶层的 rope_theta
        3. 默认值 1,000,000（Qwen3 的默认值）
        """
        rope_params = getattr(self.hf_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            return rope_params.get("rope_theta", rope_params.get("base", 1_000_000.0))
        return getattr(self.hf_config, "rope_theta", 1_000_000.0)

    @property
    def rope_scaling(self):
        """
        RoPE 扩展配置（如 yarn / dynamic_ntk）。

        当前教学仓库只支持默认 RoPE，不支持任何扩展。
        如果模型配置要求扩展，会在 get_rope() 里抛出 AssertionError。
        """
        rope_params = getattr(self.hf_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            return rope_params.get("rope_scaling", None)
        return getattr(self.hf_config, "rope_scaling", None)

    # ═══════════════════════════════════════
    # dtype 处理
    # ═══════════════════════════════════════

    @property
    def dtype(self) -> str:
        """用户指定的 dtype 字符串（来自 HF config 或默认 auto）"""
        return getattr(self.hf_config, "torch_dtype", "auto")

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        模型权重和主干计算的 dtype。

        'auto' 时的优先级：bf16 > fp16 > fp32
        """
        dtype_str = self.dtype
        if dtype_str in ("bfloat16", "bf16"):
            return torch.bfloat16
        if dtype_str in ("float16", "fp16"):
            return torch.float16
        if dtype_str in ("float32", "fp32"):
            return torch.float32

        # auto：根据 GPU 能力自动选择
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
```

### 4.2 `sampling_params.py`（完善版）

```python
"""
采样参数配置

控制每条请求文本生成时的随机性和长度。

核心概念——三种采样策略的协作：
logits → (÷ temperature) → (top-k 过滤) → (top-p 过滤) → softmax → 采样

- temperature: 缩放 logits，T→0 趋近 greedy，T>1 让分布更平
- top_k: 只保留概率最高的 K 个 token，其余设为 -inf。k=0 表示不启用
- top_p (nucleus): 按概率从高到低排序，保留累积概率刚好达到 p 的最小 token 集合
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    一条请求的采样配置。

    这些字段会由 LLMEngine 复制到 Sequence 对象上。
    后续 Scheduler / ModelRunner 只读 Sequence，不回头找原始入参。
    """

    # ── 温度缩放 ──
    # temperature=0 表示 greedy（不做随机采样，直接 argmax）
    # temperature=1.0 保持原始分布
    # temperature>1.0 让分布更平（增加随机性）
    temperature: float = 1.0

    # ── Top-K 过滤 ──
    # top_k=0 表示不启用 top-k 过滤
    # top_k=50 表示只保留概率最高的 50 个 token
    top_k: int = 0

    # ── Top-P (Nucleus) 过滤 ──
    # top_p=1.0 表示不启用 nucleus sampling
    # top_p=0.9 表示只保留累积概率达到 0.9 的最小 token 集合
    top_p: float = 1.0

    # ── 长度控制 ──
    # 最多生成多少个新 token（不含 prompt）
    max_tokens: int = 4096

    # ── 终止控制 ──
    # True 表示即使遇到 EOS 也继续生成，直到 max_tokens
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        """参数边界校验"""
        # temperature >= 0：允许 0（greedy）
        assert self.temperature >= 0.0, (
            f"temperature 必须 >= 0，当前值：{self.temperature}"
        )
        # top_k >= 0：允许 0（不启用）
        assert self.top_k >= 0, (
            f"top_k 必须 >= 0，当前值：{self.top_k}"
        )
        # top_p 在 (0, 1]：因为 softmax 所有概率 > 0，top_p 必须 > 0
        assert 0.0 < self.top_p <= 1.0, (
            f"top_p 必须在 (0, 1] 内，当前值：{self.top_p}"
        )
        # max_tokens > 0：至少要生成 1 个 token
        assert self.max_tokens > 0, (
            f"max_tokens 必须 > 0，当前值：{self.max_tokens}"
        )
```

### 4.3 `engine/sequence.py`（完善版）

变化：`__init__` 中增加了 `top_k` 和 `top_p` 的复制。

```python
"""
Sequence — 用户请求的运行时状态管理

一个 Sequence 对应一条用户请求。它记录了：
- 这条请求的所有 token（prompt + 已生成的）
- 当前状态（WAITING → RUNNING → FINISHED）
- PagedAttention 的 block_table（逻辑块 → 物理页的映射表）
- 采样参数

Sequence 是调度器、BlockManager、ModelRunner 之间交换数据的核心载体。

一条请求的生命周期：
  用户输入 "Hello, how are you?"
          ↓ Tokenize
  [15496, 11, 703, 527, 499, 30]
          ↓ 创建 Sequence(status=WAITING)
  调度器分配 KV Cache → status=RUNNING, block_table=[物理页17, ...]
          ↓ Prefill: 整段 prompt 的 K/V 写入 Cache
          ↓ Decode: 逐个生成 token，追加到 token_ids
  遇到 EOS 或达到 max_tokens → status=FINISHED, 释放 KV Cache
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举

    WAITING: 新请求，尚未分配 KV Cache，在 Scheduler.waiting 队列中
    RUNNING: 已分配 KV Cache，正在生成中，在 Scheduler.running 队列中
    FINISHED: 已完成，等待返回结果给用户
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列类 —— 每条用户请求封装为一个 Sequence 对象。

    关键属性说明：
    - token_ids: 完整的 token 序列（prompt + 已生成的部分）
    - block_table: 物理块 ID 列表，例如 [17, 203, 41]
      意思是：逻辑块 0 → 物理页 17，逻辑块 1 → 物理页 203，...
      这是 PagedAttention 的核心数据结构（类比操作系统的页表）
    - num_cached_tokens: 已进入 KV Cache 的 token 数
      用于 Prefix Cache——如果前缀相同的请求，这部分不用重新计算
    """

    # ═══════════════════════════════════════
    # 类级别常量
    # ═══════════════════════════════════════
    block_size = 256           # KV Cache 块的大小（每个块存 256 个 token 的 K/V）
    counter = count()          # 全局计数器，为每条序列生成唯一 seq_id

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
    ):
        """
        初始化序列

        Args:
            token_ids: prompt 的 token ID 列表
            sampling_params: 采样参数
        """

        # ═══ 基本属性 ═══
        self.seq_id = next(Sequence.counter)  # 唯一 ID
        self.status = SequenceStatus.WAITING   # 初始状态：等待调度

        # ═══ Token 管理 ═══
        self.token_ids = copy(token_ids)       # 深拷贝（int 不可变，所以浅层也安全）
        self.last_token = token_ids[-1]        # 最后一个 token（Decode 阶段每次只取这个）
        self.num_tokens = len(self.token_ids)  # 当前总 token 数（会随生成增长）
        self.num_prompt_tokens = len(token_ids) # 原始 prompt 长度（不变）

        # ═══ PagedAttention 核心 ═══
        self.num_cached_tokens = 0             # 已写入 KV Cache 的 token 数（Prefix Cache 用）
        self.block_table = []                  # 逻辑块 → 物理块映射表，如 [17, 203, 41]

        # ═══ 采样参数（从 SamplingParams 复制，后续只读 Sequence） ═══
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # ═══════════════════════════════════════
    # Python 魔术方法
    # ═══════════════════════════════════════

    def __len__(self):
        """返回当前的 token 总数"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持切片和索引访问 token_ids"""
        return self.token_ids[key]

    def __getstate__(self):
        """
        自定义 pickle 序列化（用于多进程通信）。

        如果序列已经生成了很多 token，只传 last_token 而不是完整的 token_ids，
        减少跨进程传输开销。
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        """反序列化"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

    # ═══════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════

    @property
    def is_finished(self):
        """检查序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数 = 总 token 数 - prompt token 数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取 prompt 部分的 token（前 num_prompt_tokens 个）"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成部分的 token（从 num_prompt_tokens 开始）"""
        return self.token_ids[self.num_prompt_tokens:]

    # ═══════════════════════════════════════
    # Block 计算相关
    # ═══════════════════════════════════════

    @property
    def num_cached_blocks(self):
        """已经缓存的完整块数"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        当前需要的总块数。

        公式：ceil(num_tokens / block_size)
        例如：num_tokens=10, block_size=4 → 需要 3 个块（4+4+2）
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中的 token 数（可能不满）"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int):
        """
        获取第 i 个块的 token 列表。

        例如：token_ids=[1,2,3,4,5,6], block_size=4
        block(0) = [1,2,3,4]
        block(1) = [5,6]
        """
        assert 0 <= i < self.num_blocks, f"块索引越界：{i}"
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    # ═══════════════════════════════════════
    # 核心操作
    # ═══════════════════════════════════════

    def append_token(self, token_id: int):
        """
        追加一个新生成的 token。

        每次 Decode 阶段生成一个 token 后调用。
        会同步更新 last_token 和 num_tokens。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
```

### 4.4 `utils/context.py`（应用问题 5 的注解修复）

这一版逻辑上已经完整，唯一要动的就是 §3 问题 5 指出的两处类型注解（`max_context_len` / `max_num_blocks` 改为 `int | None`）。下面的完整代码**已包含该修复**，同时需要理解它的设计：

```python
"""
全局上下文管理 — Context

Context 用于在模型各层之间传递 PagedAttention 需要的元数据，
避免通过函数参数层层传递。

设计理由：
- Attention 层嵌套在 DecoderLayer 里，再嵌套在 Model 里
- 如果用参数传递，需要修改所有中间层的 forward 签名（28 层！）
- 全局 Context 可以「跳过」中间层，直接传递给需要的层

使用方式：
  1. ModelRunner 在 prepare_xxx() 时调用 set_context()
  2. Attention 层通过 get_context() 获取
  3. 每个 step 结束后调用 reset_context() 清理
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Context:
    """
    全局推理上下文。

    包含本轮推理所需的全部元数据。
    Prefill 和 Decode 阶段使用不同的字段子集。

    字段分为两类：
    1. Prefill 专用：cu_seqlens_q/k, max_seqlen_q/k  (FlashAttention varlen API 需要)
    2. Decode 专用：context_lens, block_tables          (FlashAttention with_kvcache API 需要)
    3. 共用：slot_mapping, kv_cache
    """

    # ═══ 阶段标识 ═══
    is_prefill: bool = False

    # ═══ Prefill 阶段参数（FlashAttention varlen API 需要） ═══
    # cu_seqlens = "cumulative sequence lengths" = 累积序列长度
    # 例如：3 条序列，长度分别为 4, 6, 11
    # cu_seqlens = [0, 4, 10, 21]  ← 注意不是 [0, 4, 7, 11]！
    # FlashAttention 用这个来知道每条序列的起止位置
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # ═══ KV Cache 写入参数（共用） ═══
    # slot_mapping[i] = token i 对应的 KV Cache 槽位号
    # 槽位号 = 物理页号 × block_size + 页内偏移
    slot_mapping: torch.Tensor | None = None

    # ═══ Decode 阶段参数 ═══
    context_lens: torch.Tensor | None = None   # [num_seqs] 每条序列的上下文长度
    block_tables: torch.Tensor | None = None   # [num_seqs, max_blocks] 所有序列的块表
    max_context_len: int | None = None         # 最大上下文长度（问题 5 修复：int | None）
    max_num_blocks: int | None = None          # 最大块数（问题 5 修复：int | None）

    # ═══ KV Cache 引用 ═══
    # [num_layers] 每层一个 tensor
    # 每个 tensor 形状: [2, num_blocks, block_size, num_kv_heads, head_dim]
    #   其中 2 表示 K 和 V 分开存储
    kv_cache: Optional[list[torch.Tensor]] = None


# ═══════════════════════════════════════
# 全局单例管理
# ═══════════════════════════════════════

# 全局单例 Context（初始为空 Context）
_current_context = Context()


def get_context() -> Context:
    """获取当前推理上下文"""
    global _current_context
    if _current_context is None:
        raise RuntimeError("Context not set. Call set_context() before model forward.")
    return _current_context


def set_context(context: Context):
    """设置当前推理上下文（ModelRunner 在准备完输入后调用）"""
    global _current_context
    _current_context = context


def reset_context():
    """
    重置上下文到空状态。

    每个 step 结束后必须调用，防止上一轮的元数据污染下一轮。
    这在 CUDA Graph replay 场景下尤其重要——静态 buffer 可能残留旧数据。
    """
    global _current_context
    _current_context = Context()


# 兼容别名
def clear_context():
    """清除上下文（reset_context 的别名）"""
    reset_context()
```

---

## 5. 验证步骤

```bash
cd nano_vll_repro

# 1. 语法检查
python -m py_compile config.py sampling_params.py engine/sequence.py utils/context.py

# 2. 快速手测 SamplingParams
python - <<'PY'
from sampling_params import SamplingParams

# temperature=0 现在应该合法
sp = SamplingParams(temperature=0.0, top_k=20, top_p=0.95)
print(f"temperature={sp.temperature}, top_k={sp.top_k}, top_p={sp.top_p}")
assert sp.temperature == 0.0
assert sp.top_k == 20
print("SamplingParams 测试通过")
PY

# 3. 快速手测 Sequence 采样参数复制
python - <<'PY'
from sampling_params import SamplingParams
from engine.sequence import Sequence

sp = SamplingParams(temperature=0.7, top_k=8, top_p=0.9, max_tokens=64)
seq = Sequence([1, 2, 3], sp)

assert seq.temperature == 0.7
assert seq.top_k == 8
assert seq.top_p == 0.9
assert seq.max_tokens == 64
print("Sequence 采样参数复制测试通过")
PY

# 4. 跑 Day1 测试
python tests/test_Day1.py
```

> **`tests/test_Day1.py` 有两个容易写错的地方，错误写法与修正写法对照如下：**
>
> **易错点 1**：`test_context()` 中 `set_context()` 的传参方式。
> `set_context()` 只接受一个 `Context` 对象，不接受裸关键字参数：
> ```python
> # 错误写法
> set_context(
>     is_prefill=True,
>     cu_seqlens_q=torch.tensor([0, 4, 6, 11], dtype=torch.int32),
>     ...
> )
>
> # 正确
> from utils.context import Context  # 确保顶部有这个导入
> set_context(Context(
>     is_prefill=True,
>     cu_seqlens_q=torch.tensor([0, 4, 6, 11], dtype=torch.int32),
>     ...
> ))
> ```
> decode 场景的调用同理，`set_context(is_prefill=False, ...)` 要写成 `set_context(Context(is_prefill=False, ...))`。
>
> **易错点 2**：`test_config()` 中 `Config()` 的参数名。
> ```python
> # 错误写法
> config = Config(model="models/Qwen3-0.6B")
>
> # 正确
> config = Config(model_path="models/Qwen3-0.6B")
> ```
> `Config` 的字段名是 `model_path`，`model` 只是一个只读 property 别名，不能作为构造参数。

预期输出：所有测试通过，特别是 `SamplingParams(temperature=0)` 不再报错。

---

## 6. 本篇你学到的核心概念

1. **Config 是「翻译层」**：把 HF 的混乱字段命名翻译成本项目的统一名字，所有 `@property` 集中做 fallback
2. **SamplingParams 控制生成的随机性**：temperature（缩放）、top_k（数量过滤）、top_p（概率累积过滤）三者的组合覆盖了从贪婪到狂野的全部采样策略
3. **Sequence 是「请求的身份证」**：token 状态 + 块表 + 采样参数全部封装在一起，各个模块通过它交换信息
4. **Context 是「全局信箱」**：绕过 28 层函数签名的修改，让 Attention 层直接读到本轮推理的元数据

---

下一篇：**Day2 — 模型组件层**（RMSNorm / SwiGLU / RoPE / 融合 Linear，逐个读透与改进）
