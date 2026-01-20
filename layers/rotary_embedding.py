"""
RoPE (Rotary Position Embedding) 旋转位置编码

RoPE 是 LLaMA、Qwen、GPT-NeoX 等模型使用的位置编码方式。
它通过旋转操作将位置信息编码到 Query 和 Key 向量中。

核心思想：
1. 将每个 token 的 Q/K 向量视为若干个 2D 向量
2. 根据 token 位置，对每个 2D 向量进行旋转
3. 旋转角度与位置成正比

优势：
1. 相对位置信息：两个 token 的注意力分数只取决于它们的相对位置
2. 可外推性：训练时未见的位置也能泛化
3. 高效：可以预计算 cos/sin 缓存
"""

from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
) -> torch.Tensor:
    """
    对输入向量应用旋转位置编码
    
    将 x 的每对相邻维度视为一个 2D 向量，应用旋转变换。
    
    数学公式（对于每对维度 (x1, x2)）:
        x1' = x1 * cos(θ) - x2 * sin(θ)
        x2' = x2 * cos(θ) + x1 * sin(θ)
    
    Args:
        x: 输入向量，形状 [..., head_dim]
        cos: 余弦值，形状与 x 兼容
        sin: 正弦值，形状与 x 兼容
    
    Returns:
        旋转后的向量，形状不变
    """
    # 尝试 Interleaved RoPE (GPT-NeoX 风格)
    # shape = x.shape
    # x_reshaped = x.view(*shape[:-1], -1, 2)
    # x1 = x_reshaped[..., 0]
    # x2 = x_reshaped[..., 1]
    # y1 = x1 * cos - x2 * sin
    # y2 = x2 * cos + x1 * sin
    # ret = torch.stack([y1, y2], dim=-1).flatten(-2)
    # return ret.to(x.dtype)
    
    # 将向量分成两半 (Standard LLaMA Style)
    # 每半维度是 head_dim // 2
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 应用 2D旋转
    y1 = x1 * cos - x2* sin
    y2 = x2 * cos + x1* sin

    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码模块
    
    在初始化时预计算所有可能位置的 cos/sin 值并缓存，
    推理时只需要查表。
    
    Args:
        head_size: 每个注意力头的维度
        rotary_dim: 应用旋转的维度（通常等于 head_size）
        max_position_embeddings: 最大位置数
        base: 频率基数（默认 10000）
    """

    def __init__(
          self,
          head_size: int,
          rotary_dim: int,
          max_position_embeddings: int,
          base: float
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "nano-vllm 要求 rotary_dim == head_size"

        # ===== 核心：计算逆频率 =====
        # 公式: inv_freq[i] = 1 / (base^(2i / rotary_dim))
        # 其中 i = 0, 1, 2, ..., rotary_dim/2 - 1
        #
        # 直观理解：
        # - 低维度 (i 小): 频率高，捕获近距离位置关系
        # - 高维度 (i 大): 频率低，捕获远距离位置关系
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # ===== 计算所有位置的角度 =====
        # t = [0, 1, 3, ..., max_position - 1] 位置索引
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # freqs[pos, i] = pos * inv_freq[i]
        # 形状: [max_position, rotary_dim/2]
        freqs = torch.einsum("i, j -> ij", t, inv_freq)

        # 计算 cos 和 sin
        # 形状[max_position, rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()

        # 拼接并添加一个维度用于广播
        # 形状: [max_position, 1, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # 注册为buffer，不进行梯度计算， 随模型保存、加载
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 query 和 key 应用旋转位置编码
        
        Args:
            positions: 位置索引，形状 [num_tokens]
            query: Q 向量，形状 [num_tokens, num_heads, head_dim]
            key: K 向量，形状 [num_tokens, num_kv_heads, head_dim]
        
        Returns:
            (rotated_query, rotated_key)
        """
        # 根据位置索引查表获取对应的cos/sin
        # cos_sin 形状：[num_tokens, 1, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]

        # 分割 cos 和 sin
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 应用旋转
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        return query, key

@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None
) -> RotaryEmbedding:
    """
    获取RoPE实例的工厂函数

    Args:
        head_size:
        rotary_dim:
        max_position:
        base:
        rope_scaling: 位置插值
    """
    assert rope_scaling is None, "nano-vllm 不支持 位置插值/外推配置"

    return RotaryEmbedding(head_size, rotary_dim, max_position, base)