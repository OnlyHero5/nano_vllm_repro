"""Qwen3 模型实现（简化版 - Day 2）

这是不含 PagedAttention 的简化版本，用于验证模型架构。
Day 3 将添加完整的 KV Cache 支持。

模型架构:
Qwen3ForCausalLM
├── Qwen3Model
│   ├── Embedding
│   ├── DecoderLayer × N
│   │   ├── RMSNorm (input)
│   │   ├── Attention (Q/K/V/O + RoPE)
│   │   ├── RMSNorm (post)
│   │   └── MLP (gate_up + down)
│   └── RMSNorm (final)
└── LM Head
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig

from layers.activation import SiluAndMul
from layers.layernorm import RMSNorm
from layers.rotary_embedding import get_rope
from layers.attention import Attention
from layers.linear import QKVLinear, MergedLinear, RowLinear

class Qwen3Attention(nn.Module):
    """Qwen3 注意力层
    
    特点:
    - Grouped Query Attention (GQA): num_kv_heads < num_heads
    - Q/K Norm（Qwen3 特有，始终启用）
    - RoPE 位置编码
    - PagedAttention 支持
    """
     
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int | None = None,
            max_position: int = 4096 * 32,
            rms_norm_eps: float = 1e-6,
            qkv_bias: bool = False,
            rope_theta: float = 1000000.0,
            layer_idx: int = 0
    ) -> None:
        super().__init__()

        # ===== 头数配置 =====
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads

        # Q K V 维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 缩放因子
        self.scaling = self.head_dim ** (-0.5)

        # ===== 投影层 =====
        self.qkv_proj = QKVLinear(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bias=qkv_bias
        )
        self.o_proj = RowLinear(
            self.q_size,
            hidden_size,
            bias=False
        )

        # ===== RoPE =====
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta
        )

        # =====  Q/K Norm（Qwen3 特有） =====
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # ===== PagedAttention =====
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_idx=layer_idx,
        )


    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            positions: [num_tokens] 位置索引
            hidden_states: [num_tokens, hidden_size]
        
        Returns:
            [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]

        # ===== Q K V 投影 =====
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 重塑为多头形式：[num_tokens, hidden] -> [num_tokens, num_heads, head_dim]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # =====  Q / K Norm =====
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # ===== RoPE 位置编码 =====
        q, k = self.rotary_emb(positions, q, k)

        # # ===== GQA：扩展KV头数以匹配Q头数
        # if self.num_kv_heads < self.num_heads:
        #     # 每个 K V 头复制多少次
        #     repeat_times = self.num_heads // self.num_kv_heads
        #     k = k.repeat_interleave(repeat_times, dim=1)
        #     v = v.repeat_interleave(repeat_times, dim=1)
        
        # # ===== 计算注意力 =====
        # # 转换为 [batch=1, num_heads, num_tokens, head_dim]
        # # TODO : flash attention实现
        # q = q.transpose(0, 1).unsqueeze(0)
        # k = k.transpose(0, 1).unsqueeze(0)
        # v = v.transpose(0, 1).unsqueeze(0)

        # # 缩放点积注意力
        # attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # # 因果掩码
        # if attention_mask is None:
        #     # 创建因果掩码
        #     attention_mask = torch.triu(
        #         torch.full((num_tokens, num_tokens), float("-inf"), device=q.device),
        #         diagonal=1
        #     )
        # attn_weights = attn_weights + attention_mask

        # # Softmax + 加权求和
        # attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.device)
        # attn_output = torch.matmul(attn_weights, v)

        # # 转换回 [num_tokens, num_heads, head_dim]
        # attn_output = attn_output.squeeze(0).transpose(0, 1)

        # ===== Attention层 =====
        attn_output = self.attn(q, k, v)
        
        # ===== 输出投影 =====
        output = self.o_proj(attn_output.reshape(num_tokens, -1))

        return output



class Qwen3MLP(nn.Module):
    """Qwen3 FFN 层
    
    使用 SwiGLU 激活函数:
    output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
    """
    
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int 
                ):
        super().__init__()

        # gate 和 up 融合成一个线性层
        self.gate_up_proj = MergedLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            num_shards=2,
            bias=False
        )
        self.down_proj = RowLinear(
            intermediate_size,
            hidden_size,
            bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # gate_up: [num_tokens, 2 * intermediate_size]
        gate_up = self.gate_up_proj(x)

        # 应用SwiGLU：[num_tokens, hidden_size]
        x = self.act_fn(gate_up)
        
        # 下投影：[num_tokens, hidden_size]
        x = self.down_proj(x)

        return x



class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Decoder 层
    
    Pre-Norm 架构:
    x → RMSNorm → Attention → + (残差)
                              ↓
                           RMSNorm → MLP → + (残差)
    """
    def __init__(self, config, layer_idx: int = 0) -> None:
        super().__init__()

        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', None),
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            rope_theta=getattr(config, 'rope_theta', 100_0000.0),
            layer_idx=layer_idx
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: 位置索引
            hidden_states: 输入
            residual: 残差（首层为 None）
            attention_mask: 因果掩码
        
        Returns:
            (output, residual)
        """

        # ===== Pre-Norm + Attention =====
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)

        # ===== Post-Norm + MLP =====
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
    


class Qwen3Model(nn.Module):
    """
    Qwen3 主模型 （不含LM head）
    """
    def __init__(
            self,
            config
    ) -> None:
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config,layer_idx=i) for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [num_tokens] token ID
            positions: [num_tokens] 位置索引， 若为 None 则自动生成
        
        Returns:
            [num_tokens, hidden_size]
        """
        # 自动生成位置索引
        if positions is None:
            positions = torch.arange(len(input_ids), device=input_ids.device)
        
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)

        # 逐层处理
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 最终归一化
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states



class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型 （完整模型）
    """

    # ====== 融合权重映射模型 =====
    packed_modules_mapping = {
        # Q K V 融合部分
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # Gate-UP 融合
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1)
    }


    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # embedding层和 lm_head层权重共享
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回logits"""
        hidden_states = self.model(input_ids, positions)
        logits = self.lm_head(hidden_states)
        return logits
    
    @classmethod
    def from_pretrained(cls, mode_path: str):
        config = AutoConfig.from_pretrained(mode_path)
        model = cls(config)

        # TODO: 完整映射
        print(f"[Info] 模型结构已创建，权重未加载")
        print(f"[Info] hidden_size: {config.hidden_size}")
        print(f"[Info] num_layers: {config.num_hidden_layers}")
        print(f"[Info] num_heads: {config.num_attention_heads}")
        print(f"[Info] num_kv_heads: {config.num_key_value_heads}")

        return model