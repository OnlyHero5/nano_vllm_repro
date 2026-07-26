# Day 2 — 模型组件层：Transformer 的积木块

## 本篇定位

本篇拆开四个最基础的 Transformer 组件：**RMSNorm**、**SwiGLU**、**RoPE**、**融合 Linear**。

这四个模块构成了 Qwen3 模型的"积木"，它们全部放在 `layers/` 目录下，互不依赖，可以独立测试。

> **环境陷阱：`layers/__init__.py` 的 eager import**
>
> 当前 `layers/__init__.py` 长这样：
>
> ```python
> from .layernorm import RMSNorm
> from .activation import SiluAndMul
> from .rotary_embedding import RotaryEmbedding, get_rope, apply_rotary_emb
> from .attention import Attention, store_kvcache   # ← 问题在这里
> ```
>
> 第 4 行 `from .attention import ...` 会在 `import layers` 时立即触发 `attention.py` 顶部的 `import flash_attn`。后果：**没装 flash_attn 时，连 `from layers.layernorm import RMSNorm` 这种纯 torch 模块都导入失败**——因为 Python 导入任何子模块都会先执行包的 `__init__.py`。
>
> 这是"测试套件跑不起来"的最大单一原因（test_Day2 / test_Day3 整体无法收集，test_Day4 的 `test_linear_layers` / `test_sampler` 直接 `ModuleNotFoundError`），而且和代码逻辑无关。
>
> **修复方向**（二选一）：
> 1. 把 `from .attention import ...` 从 `__init__.py` 删掉，需要方直接 `from layers.attention import Attention`；
> 2. 改为惰性导入（在函数内部 import，或用 `__getattr__` 模块级延迟加载）。
>
> 本篇的四个组件（RMSNorm / SwiGLU / RoPE / 融合 Linear）本身不依赖 flash_attn。如果你想在纯 CPU 环境下跑本篇的验证脚本，先按上面方式 1 改掉 `__init__.py`，或者确保已安装 `flash-attn`。

读完本篇后，你应该能：
- 解释 RMSNorm 和 LayerNorm 的区别
- 理解 RoPE 旋转位置编码的数学直觉
- 解释"融合 Linear"为什么让推理更快
- 理解 `weight_loader` 协议如何解决"HF 权重是分离的、本地是融合的"这个矛盾

---

## 2.1 RMSNorm — 简化版 LayerNorm

### 知识点

**LayerNorm** 做两件事：减均值、除标准差。公式：

```
LayerNorm(x) = (x - mean(x)) / std(x) * γ + β
```

**RMSNorm** 只做一件事：除以均方根。公式：

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
```

**为什么大模型都用 RMSNorm？**

1. 少了均值计算，速度快 ~10%
2. 去掉 bias（β）参数，省显存
3. 实验证明效果不亚于 LayerNorm（[论文](https://arxiv.org/abs/1910.07467)）

**还有一个重要的优化**：把残差连接（`x = x + residual`）和 RMSNorm 融合成一步做，减少一次显存读写。在 Pre-Norm Transformer 中，每层都要做 `x + residual → RMSNorm`，融合后只用读一次写一次。

### 这一版长什么样

你的 `layers/layernorm.py` 实现了两个 forward 路径：

```python
# 路径1: 纯 RMSNorm
def rms_forward(self, x):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))  # 原地操作，省显存
    return x * self.weight

# 路径2: 残差 + RMSNorm 融合
def add_rms_forward(self, x, residual):
    x = x.float().add_(residual.float())  # 先做残差加法
    residual = x.to(orig_dtype)           # 保存新的残差
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    return x * self.weight, residual
```

**关键细节**：
- `torch.compile` 装饰器：告诉 PyTorch 把这个函数编译成优化后的计算图
- 中间计算用 `float()` 提升精度，最后转回 `orig_dtype`（通常是 bf16）
- `mul_()` 和 `add_()` 是原地操作（inplace），减少显存分配

### 这一层要动的地方

功能上一处都不用改——这一版 RMSNorm 立得很稳：两条路径分工清晰，精度提升和原地操作都用在了该用的地方。这一层是全书里少见的"读完即可放行"的一层。

### 完整代码

下面是带详细注释的完整版本，行为与你手上的实现一致（只顺手校正了 docstring 里的两处拼写：`redisual` → `residual`、`normalized_putput` → `normalized_output`）：

```python
# layers/layernorm.py — 完整代码

"""RMSNorm 实现

RMSNorm (Root Mean Square Layer Normalization) 是 LayerNorm 的简化版本，
被 LLaMA、Qwen 等现代大模型广泛采用。

与 LayerNorm 的区别：
- LayerNorm: 计算均值和方差，然后标准化 → y = (x - μ) / σ * γ + β
- RMSNorm: 只计算均方根 (RMS)，不计算均值 → y = x / RMS(x) * γ

为什么大模型用 RMSNorm：
1. 少算一个均值，速度快约 10%
2. 少了 β(bias) 参数，省显存
3. 实验证明效果相当（论文: Root Mean Square Layer Normalization, 2019）

本实现额外提供残差融合版本：
  传统: residual = x + residual; x = RMSNorm(residual)  # 两次显存读写
  融合: x, residual = add_rms_forward(x, residual)       # 一次显存读写
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    公式: RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

    用途：在 Transformer 的每个子层之前对输入做归一化（Pre-Norm 架构）。

    Args:
        hidden_size: 隐藏层维度（对 Qwen3-0.6B 来说 = 1024）
        eps: 防止除零的小常数，默认 1e-6
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为全1
        # 注意：RMSNorm 没有 bias 参数（与 LayerNorm 不同）
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """纯 RMSNorm（不带残差）

        用于：Embedding 后的第一次归一化，以及模型最后一层。
        因为这两个位置没有残差输入。

        计算过程：
        1. x = x² → mean → 得到每个 token 的方差
        2. x = x / sqrt(方差 + eps)  → 归一化
        3. x = x * weight            → 可学习的缩放

        Args:
            x: 形状 [num_tokens, hidden_size] 或 [batch, seq_len, hidden_size]
        Returns:
            同形状的归一化后张量
        """
        orig_dtype = x.dtype

        # 转成 FP32 做数值计算（保证精度，尤其对 bf16）
        x = x.float()

        # 计算均方值：对 hidden_size 维度求均值，keepdim=True 保持维度以便广播
        # x.pow(2) → 每个元素平方
        # .mean(dim=-1, keepdim=True) → 对最后一维求平均
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # 归一化：x = x / sqrt(var + eps)
        # torch.rsqrt(x) = 1/sqrt(x)，比先 sqrt 再除法快
        x.mul_(torch.rsqrt(var + self.eps))

        # 转回原始精度，乘以可学习的 weight
        x = x.to(orig_dtype).mul_(self.weight)

        return x

    @torch.compile
    def add_rms_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """残差 + RMSNorm 融合版本

        把「加法 + 归一化」两步合成一步，减少显存带宽消耗。
        在 Pre-Norm Transformer 的每一层都会被调用。

        传统做法：
          residual = x + residual      # 写一次显存
          x = RMSNorm(residual)        # 再读一次显存

        融合做法：
          x, residual = add_rms_forward(x, residual)
          # 只读一次、写一次，约省 50% 显存带宽

        Args:
            x: 当前子层（Attention 或 MLP）的输出
            residual: 残差连接（上一层的输出）

        Returns:
            (normalized_output, new_residual)
            - normalized_output: 归一化后的结果，送入下一个子层
            - new_residual: 更新后的残差（x + old_residual），用于下一轮
        """
        orig_dtype = x.dtype

        # 步骤1: 残差加法（转 FP32 保证精度）
        x = x.float().add_(residual.float())

        # 步骤2: 保存新的残差（下一层会用到）
        residual = x.to(orig_dtype)

        # 步骤3: RMSNorm 归一化
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """统一的 forward 入口

        根据是否传入 residual 自动选择路径：
        - residual=None → 纯 RMSNorm（首层 / 末层）
        - residual 有值  → 残差融合版本（中间层）
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

### 验证

```bash
cd nano_vll_repro
python -c "
import torch
from layers.layernorm import RMSNorm

norm = RMSNorm(1024)
x = torch.randn(4, 10, 1024)  # batch=4, seq_len=10, hidden=1024

# 纯 RMSNorm
out = norm(x)
print(f'纯 RMSNorm:  输入 {x.shape} → 输出 {out.shape}')

# 残差融合版本
residual = torch.randn_like(x)
out2, new_res = norm(x, residual)
print(f'残差融合版: 输出 {out2.shape}, 残差 {new_res.shape}')
print('RMSNorm 工作正常')
"
```

---

## 2.2 SwiGLU — 门控激活函数

### 知识点

SwiGLU 是 GLU（Gated Linear Unit）家族的一员：

```
GLU(x) = activation(gate(x)) ⊙ up(x)
```

其中 ⊙ 是逐元素乘法。不同变体用不同的 activation：
- **GLU**: sigmoid(gate) ⊙ up
- **ReGLU**: ReLU(gate) ⊙ up
- **SwiGLU**: SiLU(gate) ⊙ up  ← Qwen3/LLaMA 用的

**为什么用门控机制？**

普通的 FFN 是：
```
output = down_proj(activation(up_proj(x)))
```

SwiGLU FFN 是：
```
output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
```

多了一个 `gate_proj`（门控投影），让网络学会"哪些信息可以通过、哪些要抑制"。这类似于 LSTM 的遗忘门思想。

**SiLU** 就是 Swish 激活函数：`SiLU(x) = x * sigmoid(x)`，它比 ReLU 更平滑，梯度不会在负数区直接截断。

### 这一版长什么样

你的 `layers/activation.py` 实现很精简：

```python
class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)    # 沿 hidden 维劈成两半
        return F.silu(gate) * up          # SiLU(gate) ⊙ up
```

这里 `x` 是 `gate_up_proj` 的输出，已经拼接了 gate 和 up 两个投影的结果。`chunk(2, dim=-1)` 沿最后一维平分，前一半是 gate，后一半是 up。

### 这一版的薄弱处

没有——这一层可以原样留着。

### 完整代码

```python
# layers/activation.py — 完整代码（与当前一致）
"""SwiGLU 激活函数

SwiGLU (Swish-Gated Linear Unit) 是 GLU 家族的一员，
被 LLaMA、Qwen、PaLM 等现代大模型采用。

GLU 家族通用形式:
  GLU(x) = activation(x_gate) ⊙ x_up

常见变体:
  - GLU:    sigmoid(gate) ⊙ up
  - ReGLU:  ReLU(gate) ⊙ up
  - GEGLU:  GELU(gate) ⊙ up
  - SwiGLU: SiLU(gate) ⊙ up   ← 本项目使用

为什么用门控？普通 FFN 是 output = down(activation(up(x)))。
加了门控后变成 output = down(SiLU(gate(x)) ⊙ up(x))。
gate 学会"控制哪些信息可以通过"，效果更好。

SiLU 函数: SiLU(x) = x * sigmoid(x)，也叫 Swish。
"""
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 激活：将 gate 和 up 分离开，对 gate 应用 SiLU，然后相乘。

    输入: [num_tokens, 2 * intermediate_size]
          前半是 gate_proj 输出
          后半是 up_proj 输出
    输出: [num_tokens, intermediate_size]
    """

    def __init__(self):
        super().__init__()

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状 [*, 2*intermediate_size]，gate_up_proj 的融合输出
        Returns:
            形状 [*, intermediate_size]
        """
        # 沿最后一维切成两半
        # gate: 门控部分（将被 SiLU 激活）
        # up:   被门控部分（直接参与乘法）
        gate, up = x.chunk(2, dim=-1)

        # SiLU(gate) * up
        # F.silu(x) = x * sigmoid(x)
        return F.silu(gate) * up
```

### 验证

```bash
cd nano_vll_repro
python -c "
import torch
from layers.activation import SiluAndMul

act = SiluAndMul()
x = torch.randn(4, 10, 512)  # 2 * intermediate = 512
out = act(x)
print(f'输入 {x.shape} → 输出 {out.shape}')  # 应该变成 [4, 10, 256]
assert out.shape == (4, 10, 256)
print('SwiGLU 工作正常')
"
```

---

## 2.3 RoPE — 旋转位置编码

### 知识点

Transformer 本身是**对位置不敏感**的——同一个 token "我"出现在句首和句尾，Attention 的计算结果完全相同。所以需要位置编码告诉模型每个 token 的位置。

RoPE（Rotary Position Embedding）是 2021 年提出的方案，被 LLaMA、Qwen、GPT-NeoX 等模型采用。

**核心思想**：不是"把位置信息加到 token 向量上"，而是"把 Q 和 K 向量按位置旋转"。

数学上，对于每个相邻维度的 pair (x₁, x₂)：
```
x₁' = x₁ * cos(θ·pos) - x₂ * sin(θ·pos)
x₂' = x₂ * cos(θ·pos) + x₁ * sin(θ·pos)
```

这恰好是 2D 旋转矩阵！所以叫"旋转"位置编码。

**RoPE 的优雅之处**：两个 token 做内积时：
```
q_pos_i · k_pos_j = (通过旋转，内积只依赖于相对位置 (i-j))
```

这意味着 Attention 的分数天然包含了相对位置信息——不需要额外的"相对位置 bias"。

**频率设计**：低维度频率高（捕获近距离关系），高维度频率低（捕获远距离关系）：
```
inv_freq[i] = 1 / (base^(2i / dim))
```
`base` 通常取 10000（LLaMA）或 1000000（Qwen3-0.6B）。

### 这一版长什么样

你的实现分为三部分：

1. **`apply_rotary_emb`** — 对单个张量执行旋转操作
2. **`RotaryEmbedding`** — 预计算所有位置的 cos/sin 缓存
3. **`get_rope`** — 工厂函数，`@lru_cache` 保证相同的参数只创建一个实例

关键细节：
- `cos_sin_cache` 用 `register_buffer` 注册（不参与梯度，随模型保存/加载）
- `torch.compile` 加速 forward

### 这一版的薄弱处

`get_rope()` 当收到 `rope_scaling` 参数时应该明确报错，而不是静默忽略。当前代码只做了 `assert`，但如果模型配置了 yarn 等 RoPE 扩展，这里应该给更清晰的错误信息。

### 完整代码

```python
# layers/rotary_embedding.py — 完整代码

"""
RoPE (Rotary Position Embedding) 旋转位置编码

RoPE 是 LLaMA、Qwen、GPT-NeoX 等模型使用的位置编码方式。
它通过旋转操作将位置信息编码到 Query 和 Key 向量中。

核心思想：
1. 将每个 token 的 Q/K 向量视为若干对 2D 向量 (x₁, x₂), (x₃, x₄), ...
2. 根据 token 位置，对每对 2D 向量进行旋转
3. 旋转角度与位置成正比：θ = pos * inv_freq[i]

旋转公式（对第 i 对维度）：
    x₁' = x₁ * cos(θ_i * pos) - x₂ * sin(θ_i * pos)
    x₂' = x₂ * cos(θ_i * pos) + x₁ * sin(θ_i * pos)

为什么这么设计？
低维度（i 小）→ 频率高 → 能区分相邻位置的细微变化（「我」和「爱你」）
高维度（i 大）→ 频率低 → 能感知远距离的位置关系（「第1句」和「第100句」）

优势：
1. 相对位置：两个 token 的注意力分数只取决于它们的相对位置
2. 可外推：训练时没见过的长度也能泛化（一定程度）
3. 高效：可以预计算 cos/sin 缓存，推理时只查表不做三角函数
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """对输入向量应用旋转位置编码

    将 x 沿最后一维分成两半 (x₁, x₂)，然后对每对执行旋转：
        x₁' = x₁ * cos - x₂ * sin
        x₂' = x₂ * cos + x₁ * sin

    Args:
        x: 输入向量，形状 [..., head_dim]。通常 head_dim=64 或 128
        cos: 预计算的余弦值，形状与 x 兼容（通过广播）
        sin: 预计算的正弦值

    Returns:
        旋转后的向量，形状不变
    """
    # 沿最后一维切成两半：前半 x₁，后半 x₂
    # 每半维度 = head_dim // 2
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 应用 2D 旋转
    y1 = x1 * cos - x2 * sin  # x₁' = x₁·cos - x₂·sin
    y2 = x2 * cos + x1 * sin  # x₂' = x₂·cos + x₁·sin

    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码模块

    在初始化时预计算所有可能位置的 cos/sin 值并缓存。
    推理时只需根据 positions 查表，不需要算三角函数。

    Args:
        head_size: 每个注意力头的维度（如 64 或 128）
        rotary_dim: 应用旋转的维度（通常等于 head_size）
        max_position_embeddings: 最大位置数（如 4096）
        base: 频率基数，默认 10000（LLaMA 风格），Qwen3 用 1000000
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size

        # nano-vLLM 简化：只支持 rotary_dim == head_size
        # 某些模型（如 GPT-NeoX）可能只对部分维度做旋转
        assert rotary_dim == head_size, "nano-vllm 要求 rotary_dim == head_size"

        # ===== 核心：计算逆频率 =====
        # 公式: inv_freq[i] = 1 / (base^(2i / rotary_dim))
        # 其中 i = 0, 1, 2, ..., rotary_dim/2 - 1
        #
        # 频率分布：
        #   i=0: inv_freq = 1/(base^0) = 1.0          → 最高频率
        #   i=dim/2-1: inv_freq = 1/(base^1) ≈ 1/base → 最低频率
        #
        # 直观理解：
        #   高频 = 旋转快 = 能区分"我"和"爱"（相邻位置）
        #   低频 = 旋转慢 = 能感知"第1句"和"第100句"（远距离）
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )

        # ===== 预计算所有位置的 cos/sin =====
        # t: [0, 1, 2, ..., max_position - 1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # freqs[i, j] = t[i] * inv_freq[j]  → 每个位置每个频率的角度
        # 形状: [max_position, rotary_dim/2]
        freqs = torch.einsum("i, j -> ij", t, inv_freq)

        # 计算 cos 和 sin
        cos = freqs.cos()  # [max_position, rotary_dim/2]
        sin = freqs.sin()  # [max_position, rotary_dim/2]

        # 拼接 cos 和 sin：[max_position, rotary_dim]
        # 然后加一个 batch 维度：[max_position, 1, rotary_dim]
        # 这个 1 是为了和 Q/K 的 head 维度做广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # register_buffer: 不参与梯度，随模型保存/加载到 GPU
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 query 和 key 应用旋转位置编码

        不对 value 做 RoPE！只有 Q 和 K 需要位置信息（Attention 的内积）。

        Args:
            positions: 每个 token 的绝对位置，形状 [num_tokens]
            query: Q 向量，形状 [num_tokens, num_heads, head_dim]
            key: K 向量，形状 [num_tokens, num_kv_heads, head_dim]

        Returns:
            (rotated_query, rotated_key)，形状不变
        """
        # 根据位置索引查表获取对应的 cos/sin
        # cos_sin_cache 形状 [max_position, 1, rotary_dim]
        # positions 索引后 → [num_tokens, 1, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]

        # 沿最后一维切成 cos 和 sin
        # 每半: [num_tokens, 1, rotary_dim/2]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 对 Q 和 K 分别应用旋转
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key


@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """获取 RoPE 实例的工厂函数

    @lru_cache 保证相同参数只创建一个实例（避免重复初始化）。

    Args:
        head_size: 每个头的维度
        rotary_dim: 旋转维度（通常 = head_size）
        max_position: 最大位置数
        base: 频率基数
        rope_scaling: 位置插值/外推配置（如 yarn、dynamic_ntk）。
                     当前 nano-vLLM 教学版不支持，传入非 None 会报错。

    Raises:
        AssertionError: 如果传入 rope_scaling（教学版不支持 RoPE 扩展）
    """
    if rope_scaling is not None:
        raise AssertionError(
            f"当前 nano-vLLM 教学版不支持 RoPE 扩展（rope_scaling={rope_scaling}）。"
            f"如果你下载的模型配置了 yarn / dynamic_ntk 等扩展方案，"
            f"请参考 HuggingFace transformers 的 RoPE 实现来添加支持。"
        )

    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
```

### 验证

```bash
cd nano_vll_repro
python -c "
import torch
from layers.rotary_embedding import get_rope

# 创建 RoPE
rope = get_rope(head_size=64, rotary_dim=64, max_position=1024, base=10000)

# 模拟 5 个 token 的 Q 和 K
positions = torch.arange(5)
q = torch.randn(5, 4, 64)  # 5 tokens, 4 Q heads, 64 dim
k = torch.randn(5, 2, 64)  # 5 tokens, 2 KV heads, 64 dim

q_rot, k_rot = rope(positions, q, k)
print(f'Q 旋转前后: {q.shape} → {q_rot.shape}')
print(f'K 旋转前后: {k.shape} → {k_rot.shape}')

# 验证：旋转不改变模长
q_norm_before = q.norm(dim=-1).mean()
q_norm_after = q_rot.norm(dim=-1).mean()
print(f'Q 模长变化: {q_norm_before:.4f} → {q_norm_after:.4f} (应几乎不变)')
assert torch.allclose(q_norm_before, q_norm_after, atol=1e-4)
print('RoPE 工作正常')
"
```

---

## 2.4 融合 Linear — 让推理更快的核心设计

### 知识点

**问题**：HuggingFace 的 Qwen3 权重是分离存储的：
```
model.layers.0.self_attn.q_proj.weight  # [hidden, hidden]
model.layers.0.self_attn.k_proj.weight  # [kv_hidden, hidden]
model.layers.0.self_attn.v_proj.weight  # [kv_hidden, hidden]
```

如果推理时分别做三次矩阵乘法：
```python
q = x @ W_q.T  # (n, h) @ (h, h_out_q) → (n, h_out_q)
k = x @ W_k.T  # (n, h) @ (h, h_out_k) → (n, h_out_k)
v = x @ W_v.T  # (n, h) @ (h, h_out_v) → (n, h_out_v)
```

**融合做法**：把三个权重矩阵拼成一个大矩阵：
```
W_qkv = [W_q | W_k | W_v].T  # (h_out_q + h_out_k + h_out_v, h)
qkv = x @ W_qkv.T             # 一次矩阵乘法完成 Q/K/V
q, k, v = qkv.split(...)      # 再按尺寸拆分
```

**为什么更快？** GPU 的矩阵乘法（GEMM）效率很高。一次大矩阵乘法比三次小矩阵乘法更快，因为减少了 kernel launch 开销和数据搬运次数。

**同样道理**，`gate_proj` 和 `up_proj` 也融合成 `gate_up_proj`。

### weight_loader 协议

融合带来的新问题：HF 权重是分开的，本地是融合的。怎么加载？

**方案**：给每个融合参数绑定一个 `weight_loader` 函数，告诉它"怎么把分离的权重写入融合张量的正确位置"。

```python
# layers/linear.py
self.weight.weight_loader = self._weight_loader  # 绑定方法到参数上

# utils/loader.py
for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
    if original_name in weight_name:
        param = model.get_parameter(packed_name)
        param.weight_loader(param, loaded_weight, shard_id)  # 调参数的专属 loader
```

`packed_modules_mapping` 定义了映射规则：
```python
{
    "q_proj": ("qkv_proj", "q"),     # HF 的 q_proj → 本地的 qkv_proj，写入 Q 区间
    "k_proj": ("qkv_proj", "k"),     # HF 的 k_proj → 本地的 qkv_proj，写入 K 区间
    "v_proj": ("qkv_proj", "v"),     # HF 的 v_proj → 本地的 qkv_proj，写入 V 区间
    "gate_proj": ("gate_up_proj", 0),  # HF 的 gate_proj → 本地的 gate_up_proj，shard 0
    "up_proj": ("gate_up_proj", 1),    # HF 的 up_proj → 本地的 gate_up_proj，shard 1
}
```

这正是 loader.py 能"不理解内部布局"的关键——它只需要根据 mapping 把权重分发给对应的 weight_loader。

### 这一版长什么样

你的 `layers/linear.py` 实现了三种融合 Linear：

| 类 | 用途 | 融合了什么 | 布局 |
|-----|------|-----------|------|
| `QKVLinear` | Attention 的 QKV 投影 | q_proj + k_proj + v_proj | [Q | K | V] |
| `MergedLinear` | FFN 的 gate+up 投影 | gate_proj + up_proj | [gate | up] |
| `RowLinear` | o_proj 和 down_proj | 无融合（单卡版就是普通 Linear） | 普通 |

### 这一版的薄弱处

1. **`QKVLinear` 只给 `weight` 绑了 `weight_loader`，没有给 `bias` 绑定**。如果模型有 bias（实际 Qwen3-0.6B 没有，但写代码要考虑健壮性），bias 不会被加载。
2. **`default_weight_loader` 缺少 dtype/device 对齐**。如果 safetensors 是 fp32 存 CPU、模型参数是 bf16 存 GPU，直接 `copy_` 可能报错或不精确。
3. **编译优化**：`forward` 可以用 `torch.compile` 加速。

### 完整代码（修复后）

```python
# layers/linear.py — 完整代码（修复版）

"""融合 Linear 层

解决 HuggingFace 权重分离和本地权重融合之间的"翻译"问题。

为什么需要自定义 Linear？
1. HuggingFace 的权重是分离的（q_proj, k_proj, v_proj）
2. 我们的模型是融合的（qkv_proj），一次矩阵乘法完成 Q/K/V
3. 需要 weight_loader 协议来正确"拼接"权重

GPU 上的直觉：
  分离: x @ W_q, x @ W_k, x @ W_v    → 3 次 kernel launch
  融合: x @ [W_q | W_k | W_v]        → 1 次 kernel launch (更快！)

关键设计：
- QKVLinear 保存 num_heads, num_kv_heads, head_dim，用于计算 Q/K/V 的尺寸
- 通过 weight_loader 方法按 shard_id 写入正确位置
- RowLinear 目前是普通 Linear，为未来 Tensor Parallel 预留接口
"""
import torch
from torch import nn


def copy_weight_to_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """安全地将从 safetensors 读到的权重写入目标参数。

    为什么不能直接 param.data.copy_(loaded_weight)？
    1. loaded_weight 存在 CPU 上（safetensors 默认），param 可能在 GPU 上
    2. loaded_weight 可能是 fp32，param 可能是 bf16/fp16
    3. 不先对齐 device/dtype 会报错或产生静默精度损失

    这个函数只有两行，但它解决了 LLM 权重加载中最常见的 bug。
    """
    loaded_weight = loaded_weight.to(device=param.device, dtype=param.dtype)
    param.data.copy_(loaded_weight)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """默认权重加载器，用于没有自定义 weight_loader 的参数。

    适用场景：embed_tokens.weight, norm.weight, lm_head.weight 等非融合参数。

    步骤：
    1. 校验形状一致（形状不匹配说明映射规则有问题）
    2. 安全复制（自动对齐 device/dtype）
    """
    assert param.data.shape == loaded_weight.shape, (
        f"Shape mismatch: param {param.data.shape} vs loaded {loaded_weight.shape}"
    )
    copy_weight_to_param(param, loaded_weight)


class QKVLinear(nn.Module):
    """QKV 融合 Linear 层

    将 Q、K、V 三个投影融合成一个矩阵乘法，提升 GPU 利用率。

    输出布局: [Q | K | V] 拼接
    - Q:   [0 : q_size]
    - K:   [q_size : q_size + kv_size]
    - V:   [q_size + kv_size : q_size + 2*kv_size]

    GQA (Grouped Query Attention) 场景（Qwen3-0.6B）：
    - num_heads = 16 (Q 头数)
    - num_kv_heads = 8 (K/V 头数，每 2 个 Q 头共享 1 个 KV 头)
    - head_dim = 128
    - q_size = 16 * 128 = 2048
    - kv_size = 8 * 128 = 1024
    - total = 2048 + 1024 + 1024 = 4096
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        # ===== 保存尺寸信息（weight_loader 需要这些来算写入区间）=====
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 计算各部分尺寸
        self.q_size = num_heads * head_dim        # Q 的维度
        self.kv_size = num_kv_heads * head_dim    # K 和 V 各自的维度
        self.total_size = self.q_size + 2 * self.kv_size

        # ===== 创建融合权重 =====
        # weight 形状: [total_size, hidden_size]
        #   行: Q段 + K段 + V段
        #   列: 输入 hidden_size
        self.weight = nn.Parameter(
            torch.empty(self.total_size, hidden_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

        # ===== 绑定 weight_loader 方法到参数上 =====
        # loader.py 会通过 param.weight_loader(param, loaded_weight, shard_id) 调用
        self.weight.weight_loader = self._weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader  # ← 修复：原来漏了这行

    def _init_weights(self):
        """Kaiming 初始化，适配 ReLU 类激活函数的方差"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ):
        """将 HF 分离的 Q/K/V 权重写入融合参数的对应区间。

        这个方法会被 loader.py 调用。

        Args:
            param: 目标参数（self.weight 或 self.bias）
            loaded_weight: HuggingFace 的分离权重（如 q_proj.weight）
            shard_id: "q" | "k" | "v" 标识写入区间

        内存布局示意（Qwen3-0.6B）:
        param.data: [4096, hidden_size]
        ├── [0:2048]    ← Q (shard_id="q")
        ├── [2048:3072] ← K (shard_id="k")
        └── [3072:4096] ← V (shard_id="v")
        """
        if shard_id == "q":
            shard_offset = 0
            shard_size = self.q_size
        elif shard_id == "k":
            shard_offset = self.q_size
            shard_size = self.kv_size
        elif shard_id == "v":
            shard_offset = self.q_size + self.kv_size
            shard_size = self.kv_size
        else:
            raise ValueError(f"Unknown shard_id: {shard_id}")

        # 先对齐 dtype/device，再写入对应区间
        target = loaded_weight.to(device=param.device, dtype=param.dtype)
        param.data[shard_offset : shard_offset + shard_size].copy_(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, hidden_size]
        Returns:
            [num_tokens, q_size + 2 * kv_size]
        """
        return nn.functional.linear(x, self.weight, self.bias)


class MergedLinear(nn.Module):
    """通用融合 Linear 层

    用于 Gate-Up 等尺寸相同的融合场景。

    SwiGLU FFN 中：
    - gate_proj: hidden → intermediate
    - up_proj:   hidden → intermediate
    - 融合后:    hidden → 2 * intermediate

    输出布局: [gate | up] 拼接
    - gate: [0 : output_size]
    - up:   [output_size : 2 * output_size]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,       # 单个分片的尺寸
        num_shards: int = 2,    # 几个分片（gate/up = 2）
        bias: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_shards = num_shards
        self.total_size = output_size * num_shards  # gate + up 的总尺寸

        # 创建融合权重
        self.weight = nn.Parameter(torch.empty(self.total_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

        # 绑定 weight_loader
        self.weight.weight_loader = self._weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader  # ← 修复：bias 也需要 loader

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int,
    ):
        """加载分片权重

        Args:
            param: 目标参数
            loaded_weight: 原始权重（gate_proj.weight 或 up_proj.weight）
            shard_id: 0=gate, 1=up
        """
        shard_offset = shard_id * self.output_size
        target = loaded_weight.to(device=param.device, dtype=param.dtype)
        param.data[shard_offset : shard_offset + self.output_size].copy_(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class RowLinear(nn.Module):
    """行并行 Linear（当前单卡版本就是普通 Linear）

    用于 o_proj 和 down_proj，输入是分片的，输出需要规约。
    单卡版本就是普通 Linear，无需切片。

    保留这个类名是为了：
    - 与未来 Tensor Parallel 中的 RowParallelLinear 保持对称
    - TP 时这里需要 all_reduce
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

        # 权重加载
        self.weight.weight_loader = self._weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接复制，无需分片（单卡版本）。

        注意用 copy_weight_to_param 而非直接 copy_：
        safetensors 通常在 CPU，模型在 GPU。
        """
        copy_weight_to_param(param, loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)
```

### 改动总结

| 改动 | 位置 | 原因 |
|------|------|------|
| 新增 `copy_weight_to_param()` | 文件顶部 | 统一处理 dtype/device 对齐，解决最常出现的权重加载 bug |
| `default_weight_loader` 改用 `copy_weight_to_param` | 文件底部 | 之前直接 `copy_`，如果 CPU→GPU 或 fp32→bf16 会出错 |
| `QKVLinear.__init__` 给 bias 绑定 loader | 第 3 节 | 之前漏了！如果模型有 bias 就加载不到 |
| `MergedLinear.__init__` 给 bias 绑定 loader | 同上 | 同理 |
| 各 loader 方法用 `to(device, dtype)` 对齐 | 各 loader | 确保 CPU→GPU、fp32→bf16 的转换 |

### 验证

```bash
cd nano_vll_repro
python -c "
import torch
from layers.linear import QKVLinear, MergedLinear, RowLinear, copy_weight_to_param

# ── 测试 QKVLinear 的 weight_loader ──
qkv = QKVLinear(hidden_size=512, num_heads=8, num_kv_heads=2, head_dim=64)

# 模拟 HF 分离权重
q_weight = torch.randn(8*64, 512)   # Q: 512 维
k_weight = torch.randn(2*64, 512)   # K: 128 维
v_weight = torch.randn(2*64, 512)   # V: 128 维

# 模拟加载过程
qkv._weight_loader(qkv.weight, q_weight, 'q')
qkv._weight_loader(qkv.weight, k_weight, 'k')
qkv._weight_loader(qkv.weight, v_weight, 'v')

# 验证：Q 段应该等于原始 q_weight
assert torch.allclose(qkv.weight.data[:512], q_weight)
assert torch.allclose(qkv.weight.data[512:640], k_weight)
assert torch.allclose(qkv.weight.data[640:], v_weight)
print('QKVLinear weight_loader 正确')

# ── 测试 MergedLinear ──
merged = MergedLinear(512, 256, num_shards=2)
gate = torch.randn(256, 512)
up = torch.randn(256, 512)
merged._weight_loader(merged.weight, gate, 0)
merged._weight_loader(merged.weight, up, 1)
assert torch.allclose(merged.weight.data[:256], gate)
assert torch.allclose(merged.weight.data[256:], up)
print('MergedLinear weight_loader 正确')

# ── 测试 forward ──
x = torch.randn(10, 512)
out = qkv(x)
assert out.shape == (10, qkv.total_size)
print(f'QKVLinear forward: {x.shape} → {out.shape}')

out = merged(x)
assert out.shape == (10, 512)
print(f'MergedLinear forward: {x.shape} → {out.shape}')

# ── 测试 dtype 对齐 ──
# 模拟：dummy param 在 CUDA 上，loaded weight 在 CPU
if torch.cuda.is_available():
    p = torch.nn.Parameter(torch.empty(4, device='cuda', dtype=torch.bfloat16))
    w = torch.ones(4, dtype=torch.float32)  # CPU, fp32
    copy_weight_to_param(p, w)
    assert p.device.type == 'cuda'
    assert p.dtype == torch.bfloat16
    print('dtype/device 对齐正确')
else:
    p = torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16))
    w = torch.ones(4, dtype=torch.float32)
    copy_weight_to_param(p, w)
    assert p.dtype == torch.bfloat16
    print('dtype 对齐正确（CPU 环境）')
"
```

---

## 5. 验证步骤

```bash
cd nano_vll_repro

# 1. 语法检查
python -m py_compile layers/layernorm.py layers/activation.py layers/rotary_embedding.py layers/linear.py

# 2. 跑 Day2 测试
python tests/test_Day2.py
```

> **`tests/test_Day2.py` 有两个典型的易错点，下面把错误写法与修正写法对照讲解：**
>
> **易错点 1**：`test_gqa()` 中给 `attn()` 多传 `attention_mask=None` 参数。
> `Qwen3Attention.forward()` 的签名是 `(self, positions, hidden_states)`，没有 `attention_mask` 参数。
> ```python
> # 错误写法
> output = attn(positions, hidden_states, attention_mask=None)
>
> # 正确
> output = attn(positions, hidden_states)
> ```
>
> **易错点 2**：`test_qwen3_model()` 若没有设置 Context 就调用 `model(input_ids)`，会崩溃在 `None[layer_idx]`。
>
> 诊断过程：`Attention.forward()` 里第一步是 `context = get_context()`。不调用 `set_context()` 时拿到的是默认的空 `Context()`——`kv_cache=None`、`slot_mapping=None`（这两个 None 只是让它跳过 `store_kvcache`，不报错），但 `is_prefill` 默认是 `False`，于是走进 `_decode_attention()`，第一行就是：
> ```python
> kv_cache = context.kv_cache[self.layer_idx]
> # TypeError: 'NoneType' object is not subscriptable
> ```
> 报错点在 decode 路径，**根因却是没设 Context**——空 Context 的 `is_prefill=False` 把一次本该是 prefill 的前向骗进了 decode 分支。
>
> 修复：前向之前设置一个 prefill Context（`kv_cache` 保持 `None` 即可，Attention 会跳过 cache 写入，走纯 FlashAttention varlen 路径），前向之后 `reset_context()`：
> ```python
> set_context(Context(
>     is_prefill=True,
>     cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda'),
>     cu_seqlens_k=torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda'),
>     max_seqlen_q=num_tokens,
>     max_seqlen_k=num_tokens,
> ))
> logits = model(input_ids)
> reset_context()
> ```
> 这也是 Day1 §1.4 讲的全局 Context 模式的反面教材：**任何直接调用模型 forward 的代码（包括测试）都必须先把 Context 摆好。**

---

## 本篇小结

本篇过了一遍 Transformer 的四个基础积木块：

| 模块 | 核心原理 | 在大模型中的角色 |
|------|---------|----------------|
| RMSNorm | 减均值不如只除 RMS，省计算 | 每层前后的归一化 |
| SwiGLU | SiLU(gate) ⊙ up，门控机制 | FFN 的激活函数 |
| RoPE | 对 Q/K 向量做位置相关旋转 | 让 Attention 感知 token 位置 |
| 融合 Linear | QKV / GateUp 多个投影拼成大矩阵一次算 | 减少 kernel launch，提升 GPU 效率 |

最重要的设计模式：**`weight_loader` 协议** — 让 loader 不理解内部布局，每个参数自己决定如何写入。这是模型加载可维护性的关键。

下一篇：**Day3 — PagedAttention 引擎**（这个项目最核心、最值得深入理解的部分）
