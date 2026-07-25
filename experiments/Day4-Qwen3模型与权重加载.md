# Day 4 — Qwen3 模型与权重加载

## 本篇目标

回顾已实现的 Qwen3 模型结构，理解每个组件如何在 Transformer 推理中协作。**关键改动**：将 `forward()` 和 `logits` 投影拆开——这是后续做 CUDA Graph 优化的基础。

完成后你将理解：
- Qwen3 的 GQA（分组查询注意力）架构
- Pre-Norm Transformer 的前向数据流
- 为什么「权重融合映射」是必不可少的桥梁
- `forward()` 返回 hidden states 而非 logits 的设计理由

---

## 1. 📖 知识点讲解

### 1.1 Qwen3 模型结构

Qwen3 是一个典型的 Pre-Norm Decoder-Only Transformer：

```
Qwen3ForCausalLM          ← 完整模型
├── Qwen3Model            ← Transformer 主干
│   ├── embed_tokens      ← 词嵌入
│   ├── DecoderLayer × 28 ← 28 层 Transformer
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── Qwen3Attention
│   │   │   ├── qkv_proj (QKVLinear)   ← Q/K/V 融合投影
│   │   │   ├── q_norm / k_norm        ← Qwen3 特有
│   │   │   ├── RoPE                   ← 旋转位置编码
│   │   │   ├── Attention              ← PagedAttention（Day3 实现的）
│   │   │   └── o_proj (RowLinear)     ← 输出投影
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   └── Qwen3MLP
│   │       ├── gate_up_proj (MergedLinear) ← gate/up 融合投影
│   │       ├── SiluAndMul                  ← SwiGLU 激活
│   │       └── down_proj (RowLinear)       ← 下投影
│   └── norm (RMSNorm)       ← 最终归一化
└── lm_head                  ← vocab 投影（hidden_size → vocab_size）
```

### 1.2 GQA（Grouped Query Attention）原理

Qwen3-0.6B 的配置：
- `num_attention_heads = 16`（Q 有 16 个头）
- `num_key_value_heads = 8`（K/V 只有 8 个头）

也就是说，**每 2 个 Q 头共享 1 组 KV 头**。

```
         Q 头                           KV 头
    ┌─────────────┐              ┌─────────────┐
    │  Q0, Q1      │────共享────▶│  KV0         │
    │  Q2, Q3      │────共享────▶│  KV1         │
    │  Q4, Q5      │────共享────▶│  KV2         │
    │  ...         │              │  ...         │
    │  Q14, Q15    │────共享────▶│  KV7         │
    └─────────────┘              └─────────────┘
```

**为什么要 GQA？**
- 减少 KV Cache 的显存占用（K/V 头数少了，cache 大小就小了）
- 推理速度更快（K/V 投影矩阵更小）
- 对生成质量影响很小（研究表明 KV 头之间的信息冗余很高）

在代码中，FlashAttention 的 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 都原生支持 GQA——你只需要传入不同数量的 Q 头和 KV 头，库会自动处理扩展。

### 1.3 Q/K Norm（Qwen3 特有）

与 LLaMA 不同，Qwen3 在 Q 和 K 投影后、RoPE 之前，对每个头做一次 RMSNorm：

```python
q = self.q_norm(q)  # 对每个 attention head 做归一化
k = self.k_norm(k)
q, k = self.rotary_emb(positions, q, k)  # 然后再做 RoPE
```

为什么要加这一步？它让 Q 和 K 的分布更稳定，训练更平滑，减少 loss spike。

### 1.4 Pre-Norm 架构

```
传统 Post-Norm:             Pre-Norm (Qwen3 使用):
x → Attention → + → Norm    x → Norm → Attention → +
                                     ↓
                                  Norm → MLP → +
```

Pre-Norm 的好处：
- 梯度流动更稳定（Norm 在子层之前）
- 训练更容易收敛
- 推理时残差路径更清晰

在代码中，我们的 `Qwen3DecoderLayer.forward()` 接收 `residual` 参数，利用 `RMSNorm.add_rms_forward()` 把「残差加法 + Norm」融合成一步，减少内存读写。

### 1.5 为什么 forward() 要返回 hidden states 而不是 logits？

当前代码（需要改）：

```python
# 当前: forward 一把梭到底
def forward(self, input_ids, positions):
    hidden_states = self.model(input_ids, positions)
    return self.lm_head(hidden_states)  # 直接返回 logits
```

改后：

```python
# 改后: forward 只返回 hidden states
def forward(self, input_ids, positions):
    return self.model(input_ids, positions)

# logits 投影独立出来
def compute_logits(self, hidden_states):
    return self.lm_head(hidden_states)
```

**为什么？** 两个原因：

1. **CUDA Graph（Day7 会学）**：lm_head 的权重矩阵是 `(151936, 1024)`——vocab 维度巨大。如果 forward 连 lm_head 一起返回，CUDA Graph 就必须把整个 lm_head 计算录进图里，图的内存占用会暴涨。拆开后，只把主干录进 graph，lm_head 留在 graph 外单独算。

2. **调试和中间处理**：ModelRunner 可以先拿到 hidden states 做一些中间处理（比如 speculative decoding 中需要对比 draft model 和 target model 的 hidden states），再单独调 `compute_logits()`。

### 1.6 权重融合映射协议

这是本项目最精巧的设计之一。回忆 Day2 的融合 Linear：

- HuggingFace 存的是：`q_proj.weight`、`k_proj.weight`、`v_proj.weight`（3 个独立矩阵）
- 我们模型存的是：`qkv_proj.weight`（1 个融合矩阵，`[Q | K | V]` 拼接）

`packed_modules_mapping` 就是告诉 loader：「HF 的 `q_proj` 对应我的 `qkv_proj`，写入时填 Q 区间（shard_id='q'）」：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),      # HF 的 q_proj → 本地的 qkv_proj，shard_id='q'
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),  # HF 的 gate_proj → 本地的 gate_up_proj，shard_id=0
    "up_proj": ("gate_up_proj", 1),
}
```

loader.py 的工作流程：

```
1. 遍历 .safetensors 文件中的所有 weight_name
2. 对每个 weight_name，检查是否包含 packed_modules_mapping 中的某个 key
3. 如果命中：
   - 把 HF 名字替换成本地名字（"q_proj" → "qkv_proj"）
   - 调用目标参数的 weight_loader(param, loaded_weight, shard_id)
4. 如果没命中：
   - 调用 default_weight_loader（直接全量复制）
```

这个设计的好处是：**loader 不关心 Q/K/V 的内部布局，每个参数自己知道怎么加载。**

---

## 2. 🔍 已有代码回顾

### models/qwen3.py（回忆关键部分）

当前代码已经完整实现了 Qwen3 的四个层级：

```
Qwen3ForCausalLM          ← 完整模型（含 lm_head 和 packed_modules_mapping）
├── Qwen3Model            ← Transformer 主干（embedding + layers + final norm）
│   ├── Qwen3DecoderLayer ← 单层（attention + MLP，Pre-Norm 架构）
│   │   ├── Qwen3Attention← 注意力层（GQA + Q/K Norm + RoPE + PagedAttention）
│   │   └── Qwen3MLP      ← 前馈网络（SwiGLU）
```

**当前 `Qwen3Attention.forward()` 的 6 步执行顺序**（注释掉的是旧的手写 attention）：

```python
# 步骤1: QKV 融合投影
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

# 步骤2: reshape 成多头
q = q.view(num_tokens, self.num_heads, self.head_dim)
k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

# 步骤3: Q/K Norm（Qwen3 特有）
q = self.q_norm(q)
k = self.k_norm(k)

# 步骤4: RoPE
q, k = self.rotary_emb(positions, q, k)

# 步骤5: Attention（走 PagedAttention 的 FlashAttention 路径）
attn_output = self.attn(q, k, v)

# 步骤6: O 投影
return self.o_proj(attn_output.reshape(num_tokens, -1))
```

### utils/loader.py（回忆关键逻辑）

loader 的核心循环：

```python
for weight_name in f.keys():
    loaded_weight = f.get_tensor(weight_name)

    # 检查是否属于融合参数
    for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
        if original_name in weight_name:
            param_name = weight_name.replace(original_name, packed_name)
            param = model.get_parameter(param_name)
            param.weight_loader(param, loaded_weight, shard_id)  # 传 shard_id
            break
    else:
        # 普通参数，直接加载
        param = model.get_parameter(weight_name)
        ...
```

这里用 Python 的 `for...else` 语法：`break` 说明命中了融合参数，走融合路径；没 `break` 说明是普通参数，走 `else` 分支。

---

## 3. ⚠️ 当前代码的问题

### 问题 1：forward() 直接返回 logits

`Qwen3ForCausalLM.forward()` 当前做两件事：主干计算 + lm_head 投影。应该拆开：

```python
# 当前（需要改）
def forward(self, input_ids, positions=None):
    hidden_states = self.model(input_ids, positions)
    logits = self.lm_head(hidden_states)
    return logits

# 改为
def forward(self, input_ids, positions=None):
    return self.model(input_ids, positions)

def compute_logits(self, hidden_states):
    return self.lm_head(hidden_states)
```

### 问题 2：ModelRunner 直接调 `self.model(input_ids, positions)` 拿 logits

这意味着 ModelRunner 依赖 forward 返回 logits。改了 forward 后，ModelRunner 也要改——但那是 Day5 的事。

### 问题 3：旧注释代码未清理

`Qwen3Attention.forward()` 中有大量被注释掉的手写 attention 代码（约 30 行），影响阅读。真实路径只有 `self.attn(q, k, v)` 一条。

### 问题 4：`from_pretrained` 参数名笔误

当前代码：
```python
# ❌ 当前（第 380 行）
def from_pretrained(cls, mode_path: str):

# ✅ 正确应为
def from_pretrained(cls, model_path: str):
```

`mode_path` 应为 `model_path`。虽然当前 `from_pretrained` 只创建模型结构不加载权重，但参数名错误会导致调用时困惑（`model` 里是模型路径，不是 "mode"）。

---

## 4. 📝 完整代码

### 4.1 `models/qwen3.py`（修改后完整版）

```python
"""
Qwen3 模型实现

模型架构（Pre-Norm Decoder-Only Transformer）:

Qwen3ForCausalLM
├── Qwen3Model
│   ├── embed_tokens (nn.Embedding)
│   ├── DecoderLayer × N
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── Qwen3Attention
│   │   │   ├── qkv_proj (QKVLinear)    ← Q/K/V 融合投影
│   │   │   ├── q_norm / k_norm         ← Qwen3 特有的 Q/K 归一化
│   │   │   ├── RoPE                    ← 旋转位置编码
│   │   │   ├── Attention               ← PagedAttention（FlashAttention + KV Cache）
│   │   │   └── o_proj (RowLinear)      ← 输出投影
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   └── Qwen3MLP
│   │       ├── gate_up_proj (MergedLinear)  ← gate/up 融合投影
│   │       ├── SiluAndMul                   ← SwiGLU 激活
│   │       └── down_proj (RowLinear)        ← 下投影
│   └── norm (RMSNorm)              ← 最终归一化
└── lm_head (nn.Linear)             ← vocab 投影

关键设计：
1. forward() 只返回 hidden states，不返回 logits
   → compute_logits() 单独做 lm_head 投影
   → 这样 CUDA Graph 可以只录主干，不录大 vocab 的 lm_head

2. packed_modules_mapping 告诉 loader 如何融合 HF 权重
   → HF: q_proj + k_proj + v_proj → 本地: qkv_proj（[Q|K|V] 拼接）
   → HF: gate_proj + up_proj      → 本地: gate_up_proj（[gate|up] 拼接）

3. Qwen3 特有的 Q/K Norm：在 RoPE 之前对每个 head 做 RMSNorm
"""
import torch
from torch import nn
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
      例如 Qwen3-0.6B: 16 个 Q 头，8 个 KV 头（每 2 个 Q 共享 1 组 KV）
    - Q/K Norm（Qwen3 特有，始终启用）：在 RoPE 之前做 RMSNorm
    - RoPE 旋转位置编码
    - PagedAttention（通过 Attention 层）
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
        self.num_heads = num_heads          # Q 头数（如 16）
        self.num_kv_heads = num_kv_heads    # KV 头数（如 8，GQA）
        self.head_dim = head_dim or hidden_size // num_heads  # 每个头的维度（如 64）

        # Q、KV 各自的输出维度
        self.q_size = self.num_heads * self.head_dim       # 16×64 = 1024
        self.kv_size = self.num_kv_heads * self.head_dim   # 8×64 = 512

        # 注意力缩放因子：1 / sqrt(head_dim)
        # 为什么要缩放？点积 Q·K^T 的方差随 head_dim 增大，缩放后使 softmax 输入在合理范围
        self.scaling = self.head_dim ** (-0.5)

        # ===== QKV 融合投影 =====
        # 一次矩阵乘法算出 Q、K、V，比三次分开算更高效
        # 输出维度 = q_size + 2*kv_size = 1024 + 512 + 512 = 2048
        self.qkv_proj = QKVLinear(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bias=qkv_bias
        )

        # ===== 输出投影 =====
        self.o_proj = RowLinear(
            self.q_size,     # 输入 = Q 的维度（num_heads × head_dim）
            hidden_size,     # 输出 = hidden_size
            bias=False
        )

        # ===== RoPE 旋转位置编码 =====
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta
        )

        # ===== Q/K Norm（Qwen3 特有）=====
        # 对每个 attention head 做 RMSNorm
        # 与 LLaMA 不同，LLaMA 没有这一步
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # ===== PagedAttention（Day3 实现的）=====
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
        Qwen3 Attention 的 6 步执行顺序：

        输入:
          positions:      [num_tokens] 每个 token 的绝对位置
          hidden_states:  [num_tokens, hidden_size]

        输出:
          [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]

        # ===== 步骤1: QKV 融合投影 =====
        # qkv 形状: [num_tokens, q_size + 2*kv_size]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # ===== 步骤2: Reshape 成多头形式 =====
        # [num_tokens, total_dim] → [num_tokens, num_heads, head_dim]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # ===== 步骤3: Q/K Norm（Qwen3 特有）=====
        # 在 RoPE 之前做，让 Q 和 K 的分布更稳定
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ===== 步骤4: RoPE 旋转位置编码 =====
        # 只旋转 Q 和 K，V 不参与
        q, k = self.rotary_emb(positions, q, k)

        # ===== 步骤5: PagedAttention =====
        # Attention 内部会从全局 Context 获取 slot_mapping/block_tables 等元数据
        # Prefill: 走 flash_attn_varlen_func（处理整段 prompt）
        # Decode:  走 flash_attn_with_kvcache（只算 1 个新 token，历史从 cache 读）
        attn_output = self.attn(q, k, v)

        # ===== 步骤6: 输出投影 =====
        # [num_tokens, num_heads, head_dim] → [num_tokens, num_heads*head_dim] → [num_tokens, hidden_size]
        output = self.o_proj(attn_output.reshape(num_tokens, -1))

        return output


class Qwen3MLP(nn.Module):
    """Qwen3 前馈网络层

    使用 SwiGLU 激活函数:
      output = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )

    其中:
    - gate_proj: hidden_size → intermediate_size
    - up_proj:   hidden_size → intermediate_size
    - down_proj: intermediate_size → hidden_size

    gate_proj 和 up_proj 融合成 gate_up_proj（一次矩阵乘法，输出 [gate | up]），
    然后 SiluAndMul 做 SiLU(gate) × up
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int
    ):
        super().__init__()

        # gate 和 up 融合成一个线性层
        # 输出维度 = 2 × intermediate_size（前半是 gate，后半是 up）
        self.gate_up_proj = MergedLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            num_shards=2,
            bias=False
        )

        # 下投影
        self.down_proj = RowLinear(
            intermediate_size,
            hidden_size,
            bias=False
        )

        # SwiGLU 激活：SiLU(gate) × up
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up: [num_tokens, 2 × intermediate_size]
        gate_up = self.gate_up_proj(x)

        # SiLU(gate) × up → [num_tokens, intermediate_size]
        x = self.act_fn(gate_up)

        # 下投影 → [num_tokens, hidden_size]
        x = self.down_proj(x)

        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Decoder 层

    Pre-Norm 架构:
        x → RMSNorm → Attention → + (残差)
                                ↓
                             RMSNorm → MLP → + (残差)

    前向传播使用融合残差：RMSNorm 内部把 x+residual 和 Norm 合并，
    减少一次内存读写往返。
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
            rope_theta=getattr(config, 'rope_theta', 1_000_000.0),
            layer_idx=layer_idx
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )

        # Pre-Norm 的 Norm 层
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
        Pre-Norm + 融合残差的前向传播

        Args:
            positions: 位置索引 [num_tokens]
            hidden_states: 输入 [num_tokens, hidden_size]
            residual: 残差连接（第一层时为 None）

        Returns:
            (output, residual): output 是当前层输出，residual 传递给下一层
        """

        # ===== Pre-Norm + Attention =====
        if residual is None:
            # 第一层：没有残差，只做 Norm
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：残差加法 + Norm 融合成一步
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # ===== Post-Norm + MLP =====
        # 同样用融合残差
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 Transformer 主干（不含 lm_head）

    结构: Embedding → DecoderLayer × N → final RMSNorm
    """

    def __init__(self, config) -> None:
        super().__init__()

        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 所有 DecoderLayer
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [num_tokens] token ID 序列
            positions: [num_tokens] 位置索引，为 None 时自动生成 0,1,2,...

        Returns:
            [num_tokens, hidden_size] 最后一层的 hidden states
        """
        # 自动生成位置索引
        if positions is None:
            positions = torch.arange(len(input_ids), device=input_ids.device)

        # 词嵌入 → [num_tokens, hidden_size]
        hidden_states = self.embed_tokens(input_ids)

        # 逐层处理（Pre-Norm + 融合残差）
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 最终归一化（注意：最后一个 residual 需要被 norm 纳入）
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 因果语言模型（完整模型）

    三层结构：
    1. Qwen3Model（Transformer 主干）
    2. lm_head（vocab 投影）
    3. packed_modules_mapping（告诉 loader 如何融合 HF 权重）

    关键设计：
    - forward() 只返回 hidden states，不返回 logits
    - compute_logits() 单独做 vocab 投影
    → 这样 CUDA Graph 可以只录制主干，不录大 vocab 的 lm_head
    """

    # ===== 融合权重映射表 =====
    # loader.py 用这个表把 HF 的分离权重融合进本地参数
    packed_modules_mapping = {
        # QKV 融合：HF 的 q_proj/k_proj/v_proj → 本地的 qkv_proj
        # shard_id "q"/"k"/"v" 告诉 weight_loader 写入区间
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        # Gate-Up 融合：HF 的 gate_proj/up_proj → 本地的 gate_up_proj
        # shard_id 0/1 告诉 weight_loader 写入前半/后半
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Transformer 主干
        self.model = Qwen3Model(config)

        # vocab 投影：hidden_size → vocab_size
        # bias=False 因为 LLM 的 lm_head 通常不带 bias
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # 权重共享：如果配置要求，embedding 和 lm_head 共享权重
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        模型前向传播 — 只返回 hidden states。

        为什么不像传统代码那样直接返回 logits？
        - ModelRunner 可能需要拿到 hidden states 做中间处理
        - CUDA Graph 不希望把 lm_head（vocab 很大）捕获进 graph
        - logits 投影由 compute_logits() 单独负责

        Returns:
            [num_tokens, hidden_size]
        """
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        将最后一层的 hidden states 投影到 vocab 维度。

        Args:
            hidden_states: [num_tokens, hidden_size]

        Returns:
            [num_tokens, vocab_size] 每个 token 对所有词表中 token 的 logits
        """
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        只创建模型结构，不加载权重。

        权重加载由 utils/loader.py 的 load_model() 负责。
        这样结构创建和权重写入不会耦合。

        Args:
            model_path: HuggingFace 模型目录路径

        Returns:
            未加载权重的模型实例
        """
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = cls(config)

        # 打印配置信息，方便确认
        print(f"[模型] 结构已创建，隐藏层维度: {config.hidden_size}")
        print(f"[模型] 层数: {config.num_hidden_layers}")
        print(f"[模型] 注意力头数: {config.num_attention_heads}")
        print(f"[模型] KV 头数: {config.num_key_value_heads}")

        return model
```

### 4.2 `utils/loader.py`（修改后完整版）

```python
"""
权重加载工具

从 HuggingFace safetensors 格式加载权重到自定义模型。

核心挑战:
1. HuggingFace 权重是分离的（q_proj, k_proj, v_proj）
2. 我们的模型是融合的（qkv_proj）
3. 需要正确映射和拼接

解决方案:
1. 模型定义 packed_modules_mapping 指定映射规则
2. 融合层的参数绑定 weight_loader 方法
3. loader 根据映射调用对应的 weight_loader
"""
import os
from glob import glob

import torch
from torch import nn

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("[警告] safetensors 未安装，请执行: pip install safetensors")

from layers.linear import default_weight_loader


def load_model(model: nn.Module, model_path: str):
    """加载模型权重

    从 HuggingFace 模型目录加载所有 .safetensors 权重文件到自定义模型。

    Args:
        model: 目标模型（需要有 packed_modules_mapping 属性）
        model_path: HuggingFace 模型目录路径

    工作流程:
    1. 获取模型的 packed_modules_mapping（如 {"q_proj": ("qkv_proj", "q"), ...}）
    2. 遍历所有 .safetensors 文件
    3. 对每个权重:
       - 检查是否属于融合参数（在 mapping 中匹配到）
       - 是: 转换名称（"q_proj"→"qkv_proj"），调用 param.weight_loader(param, weight, shard_id)
       - 否: 直接调用 default_weight_loader(param, weight) 全量复制

    权重名称示例:
      HuggingFace:     model.layers.0.self_attn.q_proj.weight
                       model.layers.0.self_attn.k_proj.weight
                       model.layers.0.self_attn.v_proj.weight
      我们的模型:       model.layers.0.self_attn.qkv_proj.weight  ← 三个融合成一个
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors 是加载权重必需的，请执行: pip install safetensors")

    # 获取融合映射表
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 统计
    loaded_count = 0
    skipped_count = 0

    # 遍历所有 .safetensors 文件（可能被分片成多个）
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"在 {model_path} 中没有找到 .safetensors 文件")

    print(f"[Loader] 发现 {len(safetensor_files)} 个权重文件")

    for file_path in safetensor_files:
        print(f"[Loader] 加载: {os.path.basename(file_path)}")

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                # 从文件读出原始权重（在 CPU 上，通常是 fp32）
                loaded_weight = f.get_tensor(weight_name)

                # ==== 检查是否属于融合参数 ====
                is_packed = False
                for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
                    if original_name in weight_name:
                        # 命中融合映射！
                        # 把 HF 名字替换成本地名字
                        # 例如: "model.layers.0.self_attn.q_proj.weight"
                        #    →  "model.layers.0.self_attn.qkv_proj.weight"
                        param_name = weight_name.replace(original_name, packed_name)

                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            print(f"[Loader] 跳过（参数不存在）: {param_name}")
                            skipped_count += 1
                            is_packed = True
                            break

                        # 调参数自己的 weight_loader（带 shard_id）
                        # 例如 QKVLinear._weight_loader 知道 shard_id="q" 写入 [0:q_size]
                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is None:
                            raise RuntimeError(
                                f"参数 {param_name} 没有 weight_loader 方法！"
                                f"融合参数必须绑定 weight_loader。"
                            )
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_count += 1
                        is_packed = True
                        break  # 已匹配到融合映射，跳出内层循环

                if not is_packed:
                    # ==== 普通参数（未命中融合映射）====
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_count += 1
                    except AttributeError:
                        # 参数在 HF 权重中存在但在我们模型中不存在（如某些可选参数）
                        skipped_count += 1

    print(f"[Loader] 完成: 加载 {loaded_count} 个权重, 跳过 {skipped_count} 个")


# 别名，保持兼容
def load_model_weights(model: nn.Module, model_path: str):
    """load_model 的别名"""
    return load_model(model, model_path)
```

---

## 5. ✅ 验证步骤

```bash
cd nano_vll_repro

# 1. 语法检查
python -m py_compile models/qwen3.py utils/loader.py

# 2. 快速验证接口（不加载权重）
python - <<'PY'
import torch
from models.qwen3 import Qwen3ForCausalLM
from transformers import AutoConfig

# 用 Qwen3-0.6B 的真实配置创建模型
config = AutoConfig.from_pretrained("models/Qwen3-0.6B", trust_remote_code=True)
model = Qwen3ForCausalLM(config)
model.eval()

# 测试 forward（应该返回 hidden states）
num_tokens = 10
input_ids = torch.randint(0, config.vocab_size, (num_tokens,))
hidden_states = model(input_ids)

print(f"输入 token 数: {num_tokens}")
print(f"Hidden states 形状: {hidden_states.shape}")  # 预期: [10, 1024]
assert hidden_states.shape == (num_tokens, config.hidden_size), "forward 应返回 hidden states"

# 测试 compute_logits
logits = model.compute_logits(hidden_states)
print(f"Logits 形状: {logits.shape}")  # 预期: [10, 151936]
assert logits.shape == (num_tokens, config.vocab_size), "compute_logits 应返回 vocab 维度"

# 验证 forward 不再返回 logits
assert hidden_states.shape[-1] == config.hidden_size, "forward 不应返回 vocab_size 维度"

print("\n✅ 接口验证通过")
PY

# 3. 加载真实权重并跑一次前向
python - <<'PY'
import torch
from models.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model

model = Qwen3ForCausalLM.from_pretrained("models/Qwen3-0.6B")
load_model(model, "models/Qwen3-0.6B")
model = model.to("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
model.eval()

# 模拟 prefill：输入 10 个 token
input_ids = torch.tensor([108386, 100168, 3837, 104840, 100346, 106725, 104198, 101925, 3837, 1773])
if torch.cuda.is_available():
    input_ids = input_ids.cuda()

with torch.inference_mode():
    hidden = model(input_ids)
    logits = model.compute_logits(hidden)
    next_token = logits[-1].argmax().item()

print(f"Hidden states 形状: {hidden.shape}")
print(f"Logits 形状: {logits.shape}")
print(f"Next token: {next_token}")
print("✅ 权重加载和前向传播正常")
PY

# 4. 跑 milestones 测试
python tests/test_Day4.py
```

> **⚠️ 现有 `tests/test_Day4.py` 有以下 bug，需要先修复再运行：**
>
> **Bug**：第 3 行硬编码了绝对路径，换台机器就跑不了。
> ```python
> # ❌ 错误（当前代码）
> sys.path.insert(0, '/home/psx/nano_vllm_repro/nano_vll_repro')
>
> # ✅ 正确
> sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
> ```
> 同时需要在文件顶部补上 `import os`。

**预期输出：**
- QKVLinear / MergedLinear weight_loader 正确写入融合权重
- Sampler greedy 模式正确
- Sequence 属性访问正常

---

## 6. 📋 Day4 检查清单

| 检查项 | 说明 |
|--------|------|
| ☐ Qwen3Attention 理解 GQA | 能解释 `num_heads=16, num_kv_heads=8` 的含义 |
| ☐ 理解 Q/K Norm 的作用 | RoPE 之前做归一化，让分布更稳定 |
| ☐ 理解 Pre-Norm 架构 | Norm 在子层之前，残差融合减少内存访问 |
| ☐ forward() → hidden states | 不再返回 logits |
| ☐ compute_logits() 独立 | lm_head 投影单独出来 |
| ☐ packed_modules_mapping | 理解 HF 分离权重 → 本地融合权重的映射协议 |
| ☐ loader.py 的 for...else | 命中了 break，没命中走 else |

---

## 7. 本篇学到的核心概念

1. **GQA 是显存和效果之间的平衡**：减少 KV 头数 → 减少 KV Cache 显存 → 对质量影响很小
2. **forward() 和 logits 投影分离是 CUDA Graph 优化前提**：graph 只录主干，大 vocab 的 lm_head 留在外面
3. **packed_modules_mapping + weight_loader 是框架设计中"关注点分离"的典范**：loader 负责路由，参数负责写入逻辑
4. **Qwen3 与 LLaMA 的关键差异**：Q/K Norm（Qwen3 有，LLaMA 没有）

下一篇：**Day5 — 调度器与 ModelRunner**（Scheduler / ModelRunner 的已有代码回顾与修复）
