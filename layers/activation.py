"""SwiGLU 激活函数实现

SwiGLU (Swish-Gated Linear Unit) 是 GLU 家族的一员，
被 LLaMA、Qwen、PaLM 等现代大模型采用。

GLU 家族的通用形式:
  GLU(x, W_gate, W_up) = activation(x @ W_gate) ⊙ (x @ W_up)

其中 ⊙ 表示逐元素乘法（门控机制）。

常见变体:
  - GLU:    σ(x @ W_gate) ⊙ (x @ W_up)     # σ = sigmoid
  - ReGLU:  ReLU(x @ W_gate) ⊙ (x @ W_up)
  - GEGLU:  GELU(x @ W_gate) ⊙ (x @ W_up)
  - SwiGLU: SiLU(x @ W_gate) ⊙ (x @ W_up)   ← 
"""
import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    """
    SwiGLU 激活函数

    输入形状是[batch, seq_len, 2*intermediate_size]

    计算过程：
        1. 将输入分成两半：gate_output 和 up_output
        2. gate_output 应用Silu激活
        3. 与up_output逐元素相乘
    """
    def __init__(self):
        super().__init__()
    
    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: 形状 [batch, seq_len, 2*intermediate_size]
                  前半部分是gate_proj输出
                  后半部分是up_proj输出
        
        Returns:
          形状 [batch, seq_len, intermidate_size]
        """
        # 沿最后一维分成两半
        # gate：门控部分
        # up：被门控部分
        gate, up = x.chunk(2, dim=-1)

        # SiLU(gate) * up
        # F.silu(x)
        return F.silu(gate) * up