# Day 11 — MoE：把 FFN 路径变成可选的专家混合

> **前置依赖**：本篇修改 `config.py`、`models/qwen3.py`、`utils/loader.py`，以主线 Day1-Day6 落地后的代码为基础。不依赖其他进阶篇。Day11A（Expert Offloading）依赖本篇。

Qwen3-0.6B 是 dense 模型：每个 token 过同一个 FFN。MoE 模型不一样——每个 token 由 router 挑出 top-k 个 expert，只过这几个 expert 的 FFN，再按 routing weight 加权合并。参数量可以大很多，但每个 token 的实际计算量反而更小。

这次把一条教学版 MoE FFN 路径接进 Qwen3 主线：

1. `config.py` 增加 MoE 配置项，默认保持 dense 行为不变。
2. `models/qwen3.py` 新增 `MoERouter`、`MoEExpert`、`Qwen3MoEMLP`。
3. `Qwen3DecoderLayer` 按配置选择 dense MLP 或 MoE MLP。
4. `utils/loader.py` 补齐 expert-aware 权重加载。
5. 新增测试，不用真实 Qwen3-MoE 权重就能验证 router、top-k、dispatch、weighted combine。

vLLM 生产级的 `FusedMoE`、expert parallel、all-to-all、Triton grouped GEMM 都不做——那些是性能层，不是第一目标。

---

## 1. 参考来源

三类真实实现，各取所需：

| 来源 | 采用什么 | 不采用什么 |
|---|---|---|
| Hugging Face `Qwen3MoeSparseMoeBlock` | router、softmax、top-k、按 expert dispatch、`index_add_` weighted combine | 不照搬完整 HF 模型文件 |
| vLLM `Qwen3MoeSparseMoeBlock` / `FusedMoE` | MoE 的语义边界：gate + experts + top-k ids/weights | 不实现 fused kernel、expert parallel、量化 kernel |
| nano-vLLM MoE PR | nano 项目里 MoE 围绕 Qwen3 decoder layer 和 expert FFN 接入 | 不要求当前仓库具备该 PR 的 fused linear 基础设施 |

参考链接：

- Hugging Face Qwen3-MoE 源码：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py>
- vLLM Qwen3-MoE 源码：<https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py>
- vLLM FusedMoE 源码：<https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py>
- nano-vLLM 主仓库：<https://github.com/GeeeekExplorer/nano-vllm>
- nano-vLLM MoE PR：<https://github.com/GeeeekExplorer/nano-vllm/pull/116>

教学边界：**单卡、语义正确、复用当前仓库已有 Linear/SwiGLU/RMSNorm 组件**。

---

## 2. 当前仓库真实状态

当前仓库里，FFN 路径在 `models/qwen3.py` 中：

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            num_shards=2,
            bias=False,
        )
        self.down_proj = RowLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

`Qwen3DecoderLayer` 里固定使用 dense MLP：

```python
self.mlp = Qwen3MLP(
    hidden_size=config.hidden_size,
    intermediate_size=config.intermediate_size,
)
```

Day11 要做的事情就是把这一处变成可选分支：

```text
Attention -> dense Qwen3MLP
Attention -> MoE Qwen3MoEMLP
```

注意：MoE 替换的是 FFN 路径，不改变 Attention、RoPE、KV cache、Scheduler、Sampler。

---

## 3. 修改 `config.py`

在 `Config` dataclass 里增加下面这些字段。默认值让当前 dense Qwen3 路径完全不变。

把字段放在 `tensor_parallel_size` 和 `enforce_eager` 之间即可：

```python
    # MoE 教学配置
    enable_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int | None = None
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
```

然后在 `__post_init__()` 加载 `hf_config` 后增加自动补齐逻辑。放在当前这一行后面：

```python
self.hf_config = AutoConfig.from_pretrained(self.model_path)
```

新增：

```python
        # MoE 配置：默认保持 dense Qwen3；如果 HF config 带 MoE 字段，则自动读取。
        hf_num_experts = getattr(self.hf_config, "num_experts", 0)
        if hf_num_experts and hf_num_experts > 0:
            self.enable_moe = True
            self.num_experts = int(hf_num_experts)
            self.num_experts_per_tok = int(getattr(self.hf_config, "num_experts_per_tok", 2))
            self.moe_intermediate_size = int(
                getattr(self.hf_config, "moe_intermediate_size", self.hf_config.intermediate_size)
            )
            self.norm_topk_prob = bool(getattr(self.hf_config, "norm_topk_prob", True))
            self.decoder_sparse_step = int(getattr(self.hf_config, "decoder_sparse_step", 1))

        if self.enable_moe:
            assert self.num_experts > 0, "enable_moe=True 时 num_experts 必须 > 0"
            assert 1 <= self.num_experts_per_tok <= self.num_experts, "num_experts_per_tok 必须在 [1, num_experts] 内"
            assert self.moe_intermediate_size is not None, "enable_moe=True 时 moe_intermediate_size 不能为空"
            assert self.decoder_sparse_step >= 1, "decoder_sparse_step 必须 >= 1"
```

这段配置做了两件事：

1. 本地 Qwen3-0.6B 仍然走 dense 路径。
2. `LLMEngine` 和 `ModelRunner` 层能知道当前请求是否启用 MoE。

还要注意当前真实构造路径：`ModelRunner._load_model()` 调用的是 `Qwen3ForCausalLM.from_pretrained(self.config.model_path)`，而 `Qwen3ForCausalLM.from_pretrained()` 会自己重新读取 Hugging Face config。因此 Day11 还必须在 `models/qwen3.py` 的 `from_pretrained()` 里补同一份 MoE 字段归一化，否则项目级 `Config` 字段不会传到 decoder layer。

---

## 4. 修改 `models/qwen3.py`

### 4.1 新增 MoE config helper 和 `is_moe_layer()` helper

放在 `Qwen3MLP` 后、`Qwen3DecoderLayer` 前：

```python
def normalize_moe_config(config):
    """把 HF Qwen3-MoE 字段归一化成本仓库 decoder layer 使用的字段。"""
    num_experts = int(getattr(config, "num_experts", 0) or 0)
    enable_moe = bool(getattr(config, "enable_moe", False) or num_experts > 0)
    config.enable_moe = enable_moe

    if not enable_moe:
        config.num_experts = 0
        config.num_experts_per_tok = 0
        config.moe_intermediate_size = None
        config.norm_topk_prob = True
        config.decoder_sparse_step = 1
        return config

    config.num_experts = num_experts
    config.num_experts_per_tok = int(getattr(config, "num_experts_per_tok", 2))
    config.moe_intermediate_size = int(
        getattr(config, "moe_intermediate_size", config.intermediate_size)
    )
    config.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
    config.decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1))

    assert config.num_experts > 0, "MoE config 必须提供 num_experts"
    assert 1 <= config.num_experts_per_tok <= config.num_experts, "num_experts_per_tok 必须在 [1, num_experts] 内"
    assert config.decoder_sparse_step >= 1, "decoder_sparse_step 必须 >= 1"
    return config


def is_moe_layer(config, layer_idx: int) -> bool:
    """判断当前 decoder layer 是否使用 MoE FFN。"""
    if not getattr(config, "enable_moe", False):
        return False

    mlp_only_layers = getattr(config, "mlp_only_layers", None)
    if mlp_only_layers is not None and layer_idx in mlp_only_layers:
        return False

    decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
    return (layer_idx + 1) % decoder_sparse_step == 0
```

这对应 Hugging Face Qwen3-MoE 的 sparse layer 选择语义：不是每一层都一定是 MoE，具体由配置决定。`normalize_moe_config()` 让普通 HF MoE config 只要包含 `num_experts` 就能启用 MoE 分支，不依赖 HF config 里原本存在 `enable_moe`。

### 4.2 新增 `MoERouter`

继续放在 `Qwen3DecoderLayer` 前：

```python
class MoERouter(nn.Module):
    """Qwen3-MoE 教学版 router。"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        assert num_experts > 0, "num_experts 必须 > 0"
        assert 1 <= top_k <= num_experts, "top_k 必须在 [1, num_experts] 内"

        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        routing_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_probs, k=self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights.to(hidden_states.dtype), topk_ids
```

不要把 router 写成 hard one-hot。Qwen3-MoE / vLLM 的主线语义都是 top-k weights + top-k ids。

### 4.3 新增 `MoEExpert`

继续放在 `Qwen3DecoderLayer` 前：

```python
class MoEExpert(nn.Module):
    """单个 expert 的 FFN。"""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            num_shards=2,
            bias=False,
        )
        self.down_proj = RowLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        return self.down_proj(hidden_states)
```

这里复用当前仓库已有 `MergedLinear`、`RowLinear`、`SiluAndMul`。这是最贴合本仓库的写法。

### 4.4 新增 `Qwen3MoEMLP`

继续放在 `Qwen3DecoderLayer` 前：

```python
class Qwen3MoEMLP(nn.Module):
    """教学版单卡 Qwen3-MoE FFN。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [MoEExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.router(hidden_states)
        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = torch.nn.functional.one_hot(
            topk_ids,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)

        active_experts = torch.where(expert_mask.sum(dim=(1, 2)) > 0)[0].tolist()
        for expert_idx in active_experts:
            topk_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_states = hidden_states[token_idx]
            current_hidden = self.experts[expert_idx](current_states)
            current_hidden = current_hidden * topk_weights[token_idx, topk_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden)

        return final_hidden_states
```

这段代码采用 expert-centric dispatch：先找每个 expert 要处理哪些 token，再对每个 expert 批量计算。它比逐 token 循环更接近 Hugging Face / vLLM / nano-vLLM MoE PR 的真实结构。

### 4.5 修改 `Qwen3DecoderLayer.__init__()`

找到当前代码：

```python
self.mlp = Qwen3MLP(
    hidden_size=config.hidden_size,
    intermediate_size=config.intermediate_size
)
```

替换成：

```python
if is_moe_layer(config, layer_idx):
    self.mlp = Qwen3MoEMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        norm_topk_prob=config.norm_topk_prob,
    )
else:
    self.mlp = Qwen3MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
    )
```

`Qwen3DecoderLayer.forward()` 不需要改。因为 dense MLP 和 MoE MLP 的接口都是：

```python
hidden_states = self.mlp(hidden_states)
```

### 4.6 修改 `Qwen3ForCausalLM.from_pretrained()`

找到当前方法：

```python
@classmethod
def from_pretrained(cls, mode_path: str):
    config = AutoConfig.from_pretrained(mode_path)
    model = cls(config)
```

替换成：

```python
@classmethod
def from_pretrained(cls, mode_path: str):
    config = AutoConfig.from_pretrained(mode_path)
    config = normalize_moe_config(config)
    model = cls(config)
```

这样 `ModelRunner._load_model()` 的现有调用方式不需要变，MoE 字段也能进入 `Qwen3DecoderLayer`。

---

## 5. 修改 `utils/loader.py`

### 5.1 当前 loader 行为（先吃透再改）

当前 `load_model()`（`utils/loader.py`）的关键片段是：

```python
packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
...
for weight_name in f.keys():
    loaded_weight = f.get_tensor(weight_name)

    is_packed = False
    for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
        if original_name in weight_name:               # 子串匹配
            param_name = weight_name.replace(original_name, packed_name)
            ...
            weight_loader(param, loaded_weight, shard_id)
            is_packed = True
            break
    if not is_packed:
        # 普通权重：直接 model.get_parameter(weight_name)
        ...
```

`Qwen3ForCausalLM.packed_modules_mapping` 只有：

```python
{
    "q_proj":     ("qkv_proj", "q"),
    "k_proj":     ("qkv_proj", "k"),
    "v_proj":     ("qkv_proj", "v"),
    "gate_proj":  ("gate_up_proj", 0),
    "up_proj":    ("gate_up_proj", 1),
}
```

把 HF Qwen3-MoE 的四类权重逐一对照一下，看它们在当前 loader 里会落到哪里：

| HF 权重名 | 子串命中 | 重写后参数名 | 在我们模型里存在吗 |
|---|---|---|---|
| `model.layers.N.mlp.experts.E.gate_proj.weight` | `gate_proj` | `model.layers.N.mlp.experts.E.gate_up_proj.weight` | 存在（`MoEExpert.gate_up_proj`） |
| `model.layers.N.mlp.experts.E.up_proj.weight` | `up_proj` | `model.layers.N.mlp.experts.E.gate_up_proj.weight` | 存在（同上，`shard_id=1`） |
| `model.layers.N.mlp.experts.E.down_proj.weight` | （无） | `model.layers.N.mlp.experts.E.down_proj.weight` | 存在（`MoEExpert.down_proj.weight`），走 `is_packed=False` 分支 |
| `model.layers.N.mlp.gate.weight`（router） | （无） | `model.layers.N.mlp.gate.weight` | 不存在！我们的对应参数叫 `model.layers.N.mlp.router.gate.weight` |

**结论：唯一硬伤是 router 的命名不匹配。**

注意当前 `load_model()` 的非 packed 分支用 `try ... except AttributeError` 把找不到的参数计入 `skipped_count` 并悄悄略过。Router 权重如果不显式重写，只会在日志里出现 "跳过" 一行，没人会立刻发现 MoE 路由没有加载。这正是必须修的原因。

### 5.2 新增 `_rewrite_moe_weight_name()`

教学版仍走 `ModuleList(MoEExpert)`，每个 expert 复用 `MergedLinear` / `RowLinear`。在 `utils/loader.py` 里、`load_model()` **函数定义之前** 新增：

```python
def _rewrite_moe_weight_name(weight_name: str) -> tuple[str, int | None]:
    """把 HF Qwen3-MoE 权重名改成本仓库 MoE 模块权重名。

    返回 (rewritten_name, moe_shard_id)：
      - 如果不是 MoE 权重，rewritten_name == weight_name 且 moe_shard_id is None。
      - router gate：rewritten_name 已重命名，shard_id=None，按普通权重加载。
      - expert gate_proj/up_proj：rewritten_name 已合并到 gate_up_proj，shard_id=0/1，
        交给 MergedLinear.weight_loader 处理（与 dense 路径完全一致）。
      - expert down_proj：rewritten_name 不变，shard_id=None，按普通权重加载。
    """
    if weight_name.endswith(".mlp.gate.weight"):
        return weight_name.replace(".mlp.gate.weight", ".mlp.router.gate.weight"), None

    if ".experts." not in weight_name:
        return weight_name, None

    if weight_name.endswith(".gate_proj.weight"):
        return weight_name.replace(".gate_proj.weight", ".gate_up_proj.weight"), 0
    if weight_name.endswith(".up_proj.weight"):
        return weight_name.replace(".up_proj.weight", ".gate_up_proj.weight"), 1
    if weight_name.endswith(".down_proj.weight"):
        return weight_name, None

    return weight_name, None
```

这里**故意**让 `_rewrite_moe_weight_name` 同时覆盖 router 和 expert 两类。Router 是必须重写的硬需求；expert 子权重虽然 packed 子串匹配也能走通，但显式重写有三个好处：

1. 把"哪些权重属于 MoE 路径"集中在一个 helper 里，方便后面扩展（例如 `n_routed_experts`、shared expert 等）。
2. 避免别人误以为 packed 子串匹配是"歪打正着"，看 5.1 的对照表就知道是必然成立的。
3. 让 loader 主循环里 MoE 权重和 dense 权重的执行路径分开，日志更直观。

### 5.3 在 `load_model()` 里**先于** packed mapping 注入 MoE 分支

在 `load_model()` 现有 `for weight_name in f.keys():` 循环里、`is_packed = False` 这一行之前，插入：

```python
                # ===== MoE 分支：必须放在 packed mapping 之前 =====
                rewritten_name, moe_shard_id = _rewrite_moe_weight_name(weight_name)
                if rewritten_name != weight_name or ".experts." in weight_name:
                    try:
                        param = model.get_parameter(rewritten_name)
                    except AttributeError:
                        print(f"[Loader] 跳过 MoE 参数不存在: {rewritten_name}")
                        skipped_count += 1
                        continue

                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if moe_shard_id is None:
                        weight_loader(param, loaded_weight)
                    else:
                        weight_loader(param, loaded_weight, moe_shard_id)
                    loaded_count += 1
                    continue
                # ===== MoE 分支结束 =====
```

注意三件事：

1. **顺序不能反**。如果把它放在 packed mapping 之后，dense `gate_proj/up_proj` 子串匹配会把 `experts.E.gate_proj/up_proj` 也吞掉——结果"碰巧能跑"，但 router 的 `mlp.gate.weight` 仍会落到非 packed 分支被静默跳过。MoE 和 dense 看似都加载成功，实际 router 是随机初始化，整模型推理结果会乱。**所以必须先于 packed mapping 注入并 `continue`。**

2. **触发条件用 `rewritten_name != weight_name or ".experts." in weight_name`**。前半段处理 router 与 expert gate/up；后半段确保 expert 的 `down_proj.weight` 即使名字没变也走 MoE 分支（这样日志/计数能明确反映 expert 加载量）。

3. **不要修改 `Qwen3ForCausalLM.packed_modules_mapping`**。它是 dense 主线的事实约定。MoE 的命名差异由 loader 自己消化，否则 dense Qwen3-0.6B 的加载路径会被意外波及。

### 5.4 校验 loader 改动没破 dense 主线

为避免 `_rewrite_moe_weight_name` 误伤普通 dense Qwen3 权重，再过一遍现有 dense 权重命中清单：

| dense 权重 | `_rewrite_moe_weight_name` 返回 | 是否进入 MoE 分支 |
|---|---|---|
| `model.layers.N.self_attn.q_proj.weight` 等 | `(weight_name, None)` | 否（不含 `.experts.`，名字也没变） |
| `model.layers.N.mlp.gate_proj.weight` | `(weight_name, None)` | 否（dense MLP 不含 `.experts.`） |
| `model.layers.N.mlp.up_proj.weight` | `(weight_name, None)` | 否 |
| `model.layers.N.mlp.down_proj.weight` | `(weight_name, None)` | 否 |
| `model.layers.N.mlp.gate.weight`（dense Qwen3 没有这条） | — | — |

dense Qwen3 不会出现 `mlp.gate.weight` 这种名字，因此 `_rewrite_moe_weight_name` 对 dense 路径**完全惰性**，回归测试 Day1–Day4 不受影响。

---

## 6. 新增 `tests/test_Day11_moe.py`

创建文件：`tests/test_Day11_moe.py`

完整内容如下：

```python
"""Day11 MoE 主线接入测试。"""

import sys
from dataclasses import dataclass

sys.path.insert(0, ".")

import torch
from torch import nn

from models.qwen3 import MoERouter, Qwen3MoEMLP, Qwen3DecoderLayer, is_moe_layer


@dataclass
class TinyConfig:
    hidden_size: int = 16
    intermediate_size: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 4
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_theta: float = 10000.0
    enable_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 24
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] | None = None


class FakeAttention(nn.Module):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_is_moe_layer_respects_enable_flag():
    config = TinyConfig(enable_moe=False)
    assert not is_moe_layer(config, 0)

    config = TinyConfig(enable_moe=True, decoder_sparse_step=2)
    assert not is_moe_layer(config, 0)
    assert is_moe_layer(config, 1)


def test_router_shapes_and_normalized_topk_weights():
    router = MoERouter(hidden_size=16, num_experts=4, top_k=2, norm_topk_prob=True)
    x = torch.randn(5, 16)
    topk_weights, topk_ids = router(x)

    assert topk_weights.shape == (5, 2)
    assert topk_ids.shape == (5, 2)
    assert topk_ids.min() >= 0
    assert topk_ids.max() < 4
    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_moe_mlp_output_shape_matches_input():
    moe = Qwen3MoEMLP(
        hidden_size=16,
        intermediate_size=24,
        num_experts=4,
        top_k=2,
        norm_topk_prob=True,
    )
    x = torch.randn(6, 16)
    y = moe(x)
    assert y.shape == x.shape


def test_moe_weighted_combine_with_fake_experts():
    class FixedRouter(nn.Module):
        def forward(self, hidden_states):
            weights = torch.tensor(
                [[0.25, 0.75], [0.50, 0.50]],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            ids = torch.tensor([[0, 1], [1, 2]], device=hidden_states.device)
            return weights, ids

    class ScaleExpert(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return x * self.scale

    moe = Qwen3MoEMLP(
        hidden_size=4,
        intermediate_size=8,
        num_experts=3,
        top_k=2,
        norm_topk_prob=True,
    )
    moe.router = FixedRouter()
    moe.experts = nn.ModuleList([ScaleExpert(1.0), ScaleExpert(2.0), ScaleExpert(4.0)])

    x = torch.ones(2, 4)
    y = moe(x)

    expected0 = x[0] * (0.25 * 1.0 + 0.75 * 2.0)
    expected1 = x[1] * (0.50 * 2.0 + 0.50 * 4.0)
    expected = torch.stack([expected0, expected1])
    assert torch.allclose(y, expected)


def test_decoder_layer_selects_moe_mlp_when_config_enables_it():
    config = TinyConfig(enable_moe=True)
    layer = Qwen3DecoderLayer(config, layer_idx=0)
    assert isinstance(layer.mlp, Qwen3MoEMLP)


def test_decoder_layer_moe_path_shape_with_fake_attention():
    config = TinyConfig(enable_moe=True)
    layer = Qwen3DecoderLayer(config, layer_idx=0)
    layer.self_attn = FakeAttention()

    positions = torch.arange(3)
    hidden_states = torch.randn(3, config.hidden_size)
    out, residual = layer(positions, hidden_states)

    assert out.shape == hidden_states.shape
    assert residual.shape == hidden_states.shape


def test_rewrite_moe_weight_name_covers_all_cases():
    from utils.loader import _rewrite_moe_weight_name

    # router gate 必须重命名（这是硬需求）
    name, shard = _rewrite_moe_weight_name("model.layers.0.mlp.gate.weight")
    assert name == "model.layers.0.mlp.router.gate.weight"
    assert shard is None

    # expert gate_proj / up_proj 合并到 gate_up_proj（与 dense MergedLinear 协议一致）
    name, shard = _rewrite_moe_weight_name("model.layers.5.mlp.experts.3.gate_proj.weight")
    assert name == "model.layers.5.mlp.experts.3.gate_up_proj.weight"
    assert shard == 0
    name, shard = _rewrite_moe_weight_name("model.layers.5.mlp.experts.3.up_proj.weight")
    assert name == "model.layers.5.mlp.experts.3.gate_up_proj.weight"
    assert shard == 1

    # expert down_proj 名字不变，但仍要由 MoE 分支统一处理
    name, shard = _rewrite_moe_weight_name("model.layers.5.mlp.experts.3.down_proj.weight")
    assert name == "model.layers.5.mlp.experts.3.down_proj.weight"
    assert shard is None

    # dense 主线权重必须保持惰性（绝不能漂移到 MoE 分支）
    for dense_name in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]:
        name, shard = _rewrite_moe_weight_name(dense_name)
        assert name == dense_name
        assert shard is None


if __name__ == "__main__":
    test_is_moe_layer_respects_enable_flag()
    test_router_shapes_and_normalized_topk_weights()
    test_moe_mlp_output_shape_matches_input()
    test_moe_weighted_combine_with_fake_experts()
    test_decoder_layer_selects_moe_mlp_when_config_enables_it()
    test_decoder_layer_moe_path_shape_with_fake_attention()
    test_rewrite_moe_weight_name_covers_all_cases()
    print("Day11 MoE tests passed")
```

这份测试故意不加载真实模型权重。它验证的是当前学习项目最该先锁住的 MoE 语义：router、top-k、expert dispatch、weighted combine、DecoderLayer 组装。

---

## 7. 验收命令

从 `nano_vll_repro/` 运行：

先跑最小 MoE smoke 命令：

```bash
python -m py_compile models/qwen3.py tests/test_Day11_moe.py
python tests/test_Day11_moe.py
```

再跑覆盖所有修改文件的全量编译：

```bash
python -m py_compile config.py models/qwen3.py utils/loader.py tests/test_Day11_moe.py
```

如果安装了 pytest，再跑：

```bash
python -m pytest tests/test_Day11_moe.py -q
```

最后跑已有基础测试，确认 dense 主线没被破坏：

```bash
python tests/test_Day1.py
python tests/test_Day2.py
python tests/test_Day3.py
python tests/test_Day4.py
```

---

## 8. 常见坑

1. **把 MoE 写成多个 MLP 平均。**
   MoE 的核心是 router 给每个 token 选择 top-k experts，然后按 routing weights 合并。

2. **把 MoE 接到 Attention 或 KV cache。**
   Day11 只替换 FFN 路径。Attention、PagedAttention、KV cache 不需要因为 MoE 第一版而改变。

3. **一开始就写 fused expert weight。**
   当前仓库已有 `MergedLinear` 和 `RowLinear`，最自然的教学路径是每个 expert 一套小 FFN。fused 3D expert weight 属于性能升级。

4. **忽略 `norm_topk_prob`。**
   Qwen3-MoE 配置里 top-k 权重是否归一化是显式语义。教学版默认 `True`，但代码要保留开关。

5. **让 MoE 默认开启。**
   `enable_moe=False` 必须保持当前 Qwen3-0.6B dense 路径完全不变。

6. **把 MoE loader 分支放在 packed mapping 之后。**
   见 §5.3 的对照表。dense `gate_proj/up_proj` 子串匹配会先吞掉 `experts.E.gate_proj/up_proj`，看似 expert 加载成功，但 router 的 `mlp.gate.weight` 因为不在 packed mapping 里，会落到非 packed 分支被静默 `skipped`。最终 router 是随机初始化，整模型推理结果会乱，且日志没明显报错。MoE 分支必须先于 packed mapping 注入并 `continue`。

7. **以为 expert 子权重不重写也能跑就不重写。**
   严格说 `experts.E.gate_proj/up_proj` 通过 packed 子串匹配确实能落到正确参数上（见 §5.1 对照表），但显式让它们走 MoE 分支是为了：(a) 把 MoE 命名差异集中在 `_rewrite_moe_weight_name` 一处，便于后续接入 shared expert / `n_routed_experts`；(b) 让 loader 日志能反映 expert 真实加载量；(c) 避免后续 dense `packed_modules_mapping` 增减时 MoE 分支被牵连。

---

## 9. 做完之后

当前仓库会多出一条清晰的 MoE FFN 路径：

```text
Qwen3DecoderLayer
  ├── Attention：保持原样
  └── MLP：
      ├── dense Qwen3MLP（默认）
      └── Qwen3MoEMLP（enable_moe=True）
```

你应该能回答三个问题：

1. MoE 为什么是 FFN 层变化，不是 attention/cache/sampler 变化。
2. router、top-k ids、top-k weights、expert compute、weighted combine 分别在哪里发生。
3. 为什么先做单卡 expert-centric dispatch，而不是直接上 vLLM 的 FusedMoE kernel。

接下来两条可选路径：

- **MoE 真上单卡**：`Day11A-MoE单卡Expert-Offloading实验篇.md`——在 Day11 的 `Qwen3MoEMLP` 上叠一层 CPU↔GPU expert hot-swap，让 8GB 单卡也能"按需加载 expert"，配合 routing 频次统计把热点 expert 钉在 GPU 上。
- **继续主线进阶**：`Day12-KV-Cache量化（int8模拟）.md`——KV cache 低精度存储（int8 量化模拟，FP8 仅认知讲解）。

两条路径互不冲突：11A 管"参数 offload"，12 管"KV 精度"，正交。
