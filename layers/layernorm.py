"""RMSNorm 实现

RMSNorm (Root Mean Square Layer Normalization) 是 LayerNorm 的简化版本，
被 LLaMA、Qwen 等现代大模型广泛采用。

与 LayerNorm 的区别：
- LayerNorm: 计算均值和方差，然后标准化
- RMSNorm: 只计算均方根 (RMS)，不计算均值

优势：
1. 计算量更少（省去均值计算）
2. 效果相当甚至更好
3. 更容易并行化
"""
import torch
from torch import nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    公式: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    
    Args:
        hidden_size: 隐藏层维度
        eps: 防止除零的小常数
    """
    def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为全1
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        RMSNorm前向传播
        
        Args:
            x: 输入张量，形状[batch,seq_len, hidden_size] 或[tokens,hidden_size]
        """
        
        orig_dtype = x.dtype

        # FP32数值运算
        x = x.float()
        # 计算每个token
        # 在hidden_size维度
        # 保持维度以便广播
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # x = x / sqrt(var+eps)
        # rsqrt = 1 / sqrt
        # mul_
        x.mul_(torch.rsqrt(var+self.eps))

        # 转回原始类型
        x = x.to(orig_dtype).mul_(self.weight)

        return x

    # @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合残差加法
        
        将residual + x 和 RMSNorm 融合成一个操作，减少内存访问。
        Pre-Norm Transformer重要优化

        Args:
            x: 当前层输入
            redisual: 残差连接的输入
        
        Returns:
            (normalized_putput, new_residual)
            - normalized_output: 归一化后的结果
            - new_residual: 更新后的残差
        """
        orig_dtype = x.dtype

        # 融合: x = x + residual 
        x = x.float().add_(residual.float())

        # 更新残差
        residual = x.to(orig_dtype)

        # 执行 RMSNorm
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var+self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
    
    