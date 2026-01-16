"""融合 Linear 层

支持 QKV 融合和 Gate-Up 融合的 Linear 层，
每个参数自带 weight_loader 方法用于权重加载。

为什么需要自定义 Linear？
1. HuggingFace 的权重是分离的（q_proj, k_proj, v_proj）
2. 我们的模型是融合的（qkv_proj）
3. 需要知道各部分的尺寸才能正确拼接

关键设计：
- QKVLinear 保存 num_heads, num_kv_heads, head_dim
- 通过 weight_loader 方法按 shard_id 写入正确位置
"""
import torch
from torch import nn



class QKVLinear(nn.Module):
    """QKV 融合 Linear 层
    
    将 Q、K、V 三个投影融合成一个矩阵乘法，提升 GPU 利用率。
    
    输出布局: [Q, K, V] 拼接
    - Q: [0 : q_size]
    - K: [q_size : q_size + kv_size]  
    - V: [q_size + kv_size : q_size + 2*kv_size]
    
    GQA (Grouped Query Attention) 场景：
    - num_heads = 16 (Q 的头数)
    - num_kv_heads = 4 (K/V 的头数，共享给多个 Q 头)
    - q_size = 16 * 64 = 1024
    - kv_size = 4 * 64 = 256
    - total = 1024 + 256 + 256 = 1536
    """
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int,
            bias: bool = False
    ):
        super().__init__()

        # =====保存尺寸信息=====
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 计算各部分尺寸
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.total_size = self.q_size + 2 * self.kv_size

        # ===== 创建融合权重 =====
        self.weight = nn.Parameter(
            torch.empty(self.total_size, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()

        # =====关键：为参数绑定 weight_loader 方法 =====
        self.weight.weight_loader = self._weight_loader
    
    def _init_weights(self): 
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: str
    ): 
        """加载 Q/K/V 分离权重到融合参数
        
        这个方法会被 loader.py 的 load_model 调用。
        
        Args:
            param: 目标参数（self.weight 或 self.bias）
            loaded_weight: HuggingFace 的分离权重（q_proj.weight 等）
            shard_id: "q" / "k" / "v" 标识写入位置
        
        内存布局示意（以 Qwen3-0.6B 为例）：
        param.data: [1536, hidden_size]
        ├── [0:1024]     <- Q (shard_id="q")
        ├── [1024:1280]  <- K (shard_id="k")  
        └── [1280:1536]  <- V (shard_id="v")
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
        
        # 写入对应区间
        param.data[shard_offset:shard_offset+shard_size].copy_(loaded_weight)

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
    
    用于 Gate-Up 等等尺寸相同的融合场景。
    
    Gate-Up 融合（SwiGLU 激活）：
    - gate_proj: hidden -> intermediate
    - up_proj: hidden -> intermediate
    - 融合后: hidden -> 2 * intermediate
    
    输出布局: [gate, up] 拼接
    - gate: [0 : intermediate_size]
    - up: [intermediate_size : 2 * intermediate_size]
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_shards: int = 2,
            bias: bool = False
    ):
        super().__init__()

        # 保存尺寸信息
        self.input_size = input_size
        self.output_size = output_size  # 单个分片的尺寸
        self.num_shards = num_shards
        self.total_size = output_size * num_shards

        # 创建融合权重
        self.weight = nn.Parameter(torch.empty(self.total_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_size))
        else:
            self.register_parameter("bias", None)

        # 初始化
        self._init_weights()

        # 绑定weight_loader
        self.weight.weight_loader = self._weight_loader
        if self.bias is not None:
            self.bias.weight_loader = self._weight_loader

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias) 

    def _weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: int
            ):
        """加载分片权重
        
        Args:
            param: 目标参数
            loaded_weight: 原始权重（gate_proj.weight 或 up_proj.weight）
            shard_id: 分片索引 (0=gate, 1=up)
        """
        shard_offset = shard_id * self.output_size
        param.data[shard_offset : shard_offset + self.output_size].copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)



class RowLinear(nn.Module):
    """行并行 Linear（单卡版本）
    
    用于 o_proj 和 down_proj，输入是分片的，输出需要规约。
    单卡版本就是普通 Linear,无需切片
    
    - 保持与 ColumnParallel 的对称性
    - 未来扩展 TP 时，这里需要 all_reduce
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter('bias', None)

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
        """直接复制，无需分片"""
        # TODO: 多卡分片支持
        param.data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)