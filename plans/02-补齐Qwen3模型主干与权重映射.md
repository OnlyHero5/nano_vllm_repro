# 02. 收口 Qwen3 模型主干和权重映射

这一篇仍然只做单卡主线。

目标是：

> 让 Qwen3 模型结构、HF 配置字段、RoPE 参数和权重映射都走同一套清楚的接口。

改完后要达到这些结果：

1. `Config` 统一暴露 Qwen3 常用字段，调用点不要到处 `getattr(config, "...")`。
2. `get_rope()` 明确说明当前只支持默认 RoPE。
3. `Qwen3Attention` 的 QKV、Q/K norm、RoPE、Attention、输出投影顺序清楚。
4. `Qwen3ForCausalLM.forward()` 返回 hidden states。
5. `compute_logits()` 单独负责 `lm_head` 投影。
6. `tests/test_Day2.py` 不再假设 `forward()` 直接返回 logits。

Tensor Parallel 的线性层签名不要在本篇引入。那是 `05` 的任务。

---

## 1. 当前代码是什么状态

### 1.1 `config.py`

当前 `Config` 能加载 HF config，但很多模型字段没有统一出口。

后面这些字段会被模型、KV Cache、TP、CUDA Graph 反复用到：

- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `rope_theta`
- `rope_scaling`
- `torch_dtype`
- `kv_torch_dtype`

如果每个调用点都自己 `getattr(...)`，代码会越来越乱。

### 1.2 `layers/rotary_embedding.py`

当前 `get_rope()` 已经有 `rope_scaling` 参数，并且会拒绝不支持的配置。

这条边界是对的。我们要做的是把说明写清楚：

> 当前教学仓库只支持默认 RoPE，不支持 yarn、longrope、dynamic_ntk 等扩展。

### 1.3 `models/qwen3.py`

当前主要问题有三个：

1. `Qwen3ForCausalLM.forward()` 直接返回 logits。
2. 没有单独的 `compute_logits()`。
3. `Qwen3Attention` 里保留了大段注释掉的手写 attention fallback，容易让读者误以为还有另一条执行路径。

后面 `ModelRunner`、CUDA Graph、benchmark 都需要清楚地区分：

```text
模型主干 forward -> hidden states
lm_head 投影 -> logits
sampler -> token
```

所以这篇要先把边界拆开。

---

## 2. 修改 `Config`

保留当前 `model_path` 作为构造参数，不要改成 `model`。

可以增加下面这些字段：

```python
@dataclass
class Config:
    # 本地模型目录。当前仓库统一使用 model_path。
    model_path: str

    # 连续批处理参数。
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096

    # 显存和优化参数。
    gpu_memory_utilization: float = 0.7
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    max_cudagraph_batch_size: int = 32

    # dtype 用字符串保存，方便从命令行或配置文件传入。
    # 真正给 torch 用时，再通过 property 转成 torch.dtype。
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"

    # 运行时填充。
    hf_config: AutoConfig | None = None
    eos: int = -1

    # KV Cache block 配置。
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
```

然后增加一组 property。它们的作用是把 HF 配置字段统一翻译成当前仓库好用的名字。

```python
@property
def hidden_size(self) -> int:
    # 后续模块统一从 Config 取 hidden_size，不直接碰 hf_config。
    return self.hf_config.hidden_size


@property
def num_attention_heads(self) -> int:
    return self.hf_config.num_attention_heads


@property
def num_key_value_heads(self) -> int:
    # 有些模型配置没有 num_key_value_heads。
    # 这种情况下退化成普通 MHA：KV 头数等于 Q 头数。
    return getattr(self.hf_config, "num_key_value_heads", self.num_attention_heads)


@property
def head_dim(self) -> int:
    # Qwen3 通常会显式给 head_dim。
    # 如果没有，就用 hidden_size // num_attention_heads 兜底。
    return getattr(
        self.hf_config,
        "head_dim",
        self.hidden_size // self.num_attention_heads,
    )


@property
def attention_bias(self) -> bool:
    return getattr(self.hf_config, "attention_bias", False)


@property
def rope_parameters(self):
    # 新版 HF 配置可能把 RoPE 参数放在 rope_parameters 里。
    return getattr(self.hf_config, "rope_parameters", None)


@property
def rope_theta(self) -> float:
    """
    当前实际使用的 RoPE base。

    读取顺序：
    1. 新式 rope_parameters
    2. 旧式 rope_theta
    3. Qwen3 常见默认值 1_000_000.0
    """
    rope_parameters = self.rope_parameters
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_theta", rope_parameters.get("base", 1_000_000.0))
    return getattr(self.hf_config, "rope_theta", 1_000_000.0)


@property
def rope_scaling(self):
    """
    这里只负责把配置传下去。

    是否支持具体 rope_scaling 策略，由 get_rope() 判断。
    不要在这里静默吞掉不支持的配置。
    """
    rope_parameters = self.rope_parameters
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_scaling", None)
    return getattr(self.hf_config, "rope_scaling", None)
```

dtype 也放到 `Config` 里统一判断：

```python
@property
def torch_dtype(self) -> torch.dtype:
    """
    模型权重和主干计算使用的 dtype。

    auto 规则：
    - CUDA 支持 bf16，就用 bf16。
    - CUDA 不支持 bf16，就用 fp16。
    - CPU 路径用 fp32。
    """
    if self.dtype == "bfloat16":
        return torch.bfloat16
    if self.dtype == "float16":
        return torch.float16
    if self.dtype == "float32":
        return torch.float32

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


@property
def kv_torch_dtype(self) -> torch.dtype:
    """
    KV Cache 的 dtype 单独配置。

    原因：
    模型主干可以用 bf16，但 KV Cache 为了省显存，可能继续用 fp16。
    """
    if self.kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if self.kv_cache_dtype == "float16":
        return torch.float16
    if self.kv_cache_dtype == "float32":
        return torch.float32
    if self.kv_cache_dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"不支持的 kv_cache_dtype: {self.kv_cache_dtype}")
```

---

## 3. 明确 `get_rope()` 的边界

`layers/rotary_embedding.py` 里的主体实现不用重写。

只需要把 `get_rope()` 写清楚：

```python
@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
) -> RotaryEmbedding:
    """
    返回当前仓库使用的 RoPE 实例。

    当前边界：
    - 支持默认 RoPE。
    - 不支持 yarn / longrope / dynamic_ntk 等扩展。

    为什么不静默忽略 rope_scaling：
    如果模型配置里要求扩展 RoPE，但本地代码偷偷当默认 RoPE 跑，
    长上下文结果会错，而且很难排查。
    """
    if rope_scaling is not None:
        raise AssertionError("当前教学仓库只支持默认 RoPE，暂不支持 rope_scaling")

    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
    )
```

---

## 4. 收口 `Qwen3Attention`

本篇继续使用当前本地类名：

- `QKVLinear`
- `MergedLinear`
- `RowLinear`

不要在这里切到 `QKVParallelLinear`。

`Qwen3Attention.forward()` 的顺序应该非常明确：

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    输入：
    - positions: 每个 token 的位置，形状 [num_tokens]
    - hidden_states: 输入隐状态，形状 [num_tokens, hidden_size]

    输出：
    - shape 仍然是 [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.shape[0]

    # 1. 一次线性层同时算出 Q、K、V。
    #    输出布局是 [Q, K, V] 拼接。
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # 2. reshape 成多头形式。
    #    q 用 num_heads，k/v 用 num_kv_heads，这是 GQA 的关键。
    q = q.view(num_tokens, self.num_heads, self.head_dim)
    k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
    v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

    # 3. Qwen3 的 q_norm / k_norm 作用在每个 head 的 head_dim 上。
    #    这一步在 RoPE 前面做。
    q = self.q_norm(q)
    k = self.k_norm(k)

    # 4. RoPE 只作用在 q/k，不作用在 v。
    q, k = self.rotary_emb(positions, q, k)

    # 5. 真正的 prefill / decode 分支在 Attention + Context 里处理。
    attn_output = self.attn(q, k, v)

    # 6. 把多头输出展平，再过 o_proj。
    output = self.o_proj(attn_output.reshape(num_tokens, -1))
    return output
```

建议删掉 `models/qwen3.py` 里大段注释掉的手写 attention fallback。

原因很简单：当前真实执行路径只有 `self.attn(q, k, v)` 这一条。

---

## 5. 拆开模型主干和 logits

`Qwen3ForCausalLM.forward()` 不要直接返回 logits。

推荐写成：

```python
class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型外壳。

    它负责三件事：
    1. 持有 Qwen3Model 主干。
    2. 持有 lm_head。
    3. 暴露 packed_modules_mapping 给 loader.py 用。
    """

    packed_modules_mapping = {
        # HF 是 q_proj/k_proj/v_proj，本地是 qkv_proj。
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),

        # HF 是 gate_proj/up_proj，本地是 gate_up_proj。
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        只返回 hidden states，不返回 logits。

        这样 ModelRunner 可以明确控制：
        主干前向 -> lm_head -> sampler。
        """
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        把 hidden states 投影到 vocab 维度。

        这个函数单独拆出来，是为了给后面的 ModelRunner、
        CUDA Graph 和 benchmark 一个清楚的边界。
        """
        return self.lm_head(hidden_states)
```

`from_pretrained()` 只创建结构，不加载权重：

```python
@classmethod
def from_pretrained(cls, model_path: str):
    """
    只创建模型结构。

    权重加载交给 utils.loader.load_model()。
    这样结构创建和权重写入不会混在一起。
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = cls(config)
    print("[Info] 模型结构已创建，权重尚未加载")
    return model
```

---

## 6. 修改 `tests/test_Day2.py`

### 6.1 模型测试要走新边界

旧逻辑通常是：

```python
logits = model(input_ids, positions)
```

新逻辑应该是：

```python
hidden_states = model(input_ids, positions)
logits = model.compute_logits(hidden_states)

assert hidden_states.shape == (num_tokens, config.hidden_size)
assert logits.shape == (num_tokens, config.vocab_size)
```

### 6.2 `Qwen3Attention` 不要传 `attention_mask`

当前 `Qwen3Attention.forward()` 的签名是：

```python
attn(positions, hidden_states)
```

测试里不要再写：

```python
attn(positions, hidden_states, attention_mask=None)
```

prefill / decode 的 mask 和 KV Cache 元数据由 `Attention + Context` 处理，不从 `Qwen3Attention.forward()` 直接传。

---

## 7. 验收命令

```bash
python -m py_compile config.py layers/rotary_embedding.py models/qwen3.py tests/test_Day2.py
python tests/test_Day2.py
```

如果只想快速看接口边界：

```bash
python - <<'PY'
import torch
from models.qwen3 import Qwen3ForCausalLM

model = Qwen3ForCausalLM.from_pretrained("models/Qwen3-0.6B")
input_ids = torch.tensor([1, 2, 3, 4])
positions = torch.arange(4)

hidden_states = model(input_ids, positions)
logits = model.compute_logits(hidden_states)

print(hidden_states.shape)
print(logits.shape)
PY
```

---

## 8. 常见坑

1. **继续让 `forward()` 直接返回 logits**
   后面 `ModelRunner` 和 CUDA Graph 都会变难拆。

2. **在 02 里提前切到 TP 线性层**
   当前 `layers/linear.py` 还不是 TP 版，提前切会让模型和算子接口对不上。

3. **收到 `rope_scaling` 但静默忽略**
   这会制造“看似兼容，实际结果不对”的假象。

4. **测试还按旧接口写**
   这会让你把测试失败误判成模型实现失败。

---

## 9. 本篇结束后你应该明白

这一篇的重点是：

1. `Config` 是 HF 配置和本地运行时之间的桥。
2. Qwen3 Attention 的执行顺序要清楚。
3. 模型主干和 logits 投影要拆开。
4. “参考上游”不等于“现在就照搬上游 TP 接口”。

下一篇进入采样：

- `03-补全Sampler与SamplingParams.md`
