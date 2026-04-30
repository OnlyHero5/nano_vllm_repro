# 11. 实现 MoE 推理主线与专家并行认知篇

这一篇开始进入一个非常容易“看懂概念、做错边界”的方向：`MoE inference`。

这一篇的目标不是把当前仓库立刻改造成完整的 Qwen3-MoE 推理系统，而是：

> 站在当前 dense Qwen3 主线的真实代码上，讲清楚从 dense 到 MoE 会发生哪些结构变化，并给出一套完整可运行的教学原型代码。

这一篇只做三件事：

1. 明确 dense Qwen3 与 MoE Qwen3 在 DecoderLayer 里的结构差异。
2. 新增一个独立实验文件，实现最小可运行的 MoE FFN 原型。
3. 说明未来如果要把它真正接进当前主线，应该改哪些边界，不应该误改哪些边界。

这一篇不做下面这些事：

- 不假装当前仓库已经原生支持 Qwen3-MoE 权重加载。
- 不修改现有 `models/qwen3.py` 主线代码。
- 不引入真实 expert parallel 分布式通信。
- 不把 Triton Group-GEMM 直接并回当前 dense 代码路径。
- 不在主线测试里伪造“MoE 已经跑通”的结论。

原因很简单：

> MoE 的核心不是“把一个 MLP 换成几个 MLP”，而是“路由、专家选择、专家计算、输出合并、并行切分”这一整套新执行语义。

---

## 1. 为什么 MoE 不能像 top-k / top-p 那样直接加一个参数就完事

当前仓库的 Qwen3 主线是 dense Transformer：

```text
Attention -> MLP
```

其中 MLP 对每个 token 都是：

> 所有隐藏维都经过同一组 FFN 权重。

MoE 则完全不同。它的核心是：

1. 先由 router 给每个 token 打分。
2. 为每个 token 选出 top-k 个 expert。
3. 每个 expert 只处理被分给自己的 token。
4. 再按 router 权重把多个 expert 输出加权合并。

也就是说：

> dense MLP 是“所有 token 过同一条 FFN 路径”，MoE 是“不同 token 动态选择不同 FFN 路径”。

所以它天然会引入四类新复杂度：

- router 打分。
- token dispatch。
- expert 计算。
- weighted combine。

如果后面再做多卡，还会多出 expert parallel 和 all-to-all / gather / scatter。

因此，这一篇的第一原则是：

> 先把 MoE 的单卡执行语义写清楚，再谈 expert parallel、group GEMM、量化和 speculative decoding 联动。

---

## 2. 当前代码是什么状态

与 MoE 最相关的当前文件有：

1. `models/qwen3.py`
2. `layers/linear.py`
3. `layers/activation.py`
4. `engine/model_runner.py`

### 2.1 当前 `Qwen3DecoderLayer` 还是标准 dense 结构

当前主线里的 FFN 部分本质上是：

```text
gate_up_proj -> SwiGLU -> down_proj
```

这是一条标准 dense MLP 路径。

### 2.2 当前 `MergedLinear` 适合做“单 expert 的 gate/up 融合”

因为在教学版 MoE 原型里，一个 expert 的内部结构仍然可以复用当前 dense MLP 的习惯：

- `MergedLinear` 负责 gate/up。
- `SiluAndMul` 负责 SwiGLU。
- `RowLinear` 负责 down projection。

### 2.3 当前主线还没有 router 与 token dispatch 概念

这也是为什么这一篇不能直接改 `models/qwen3.py` 主线。

如果强行把 router、dispatch、expert combine 写进去：

- 会立刻影响 dense Qwen3 的阅读边界。
- 会让现有 Day2 / Day4 / Day5 主线文档失真。

所以这一篇必须用新增实验文件来承接。

---

## 3. 工程边界

这一篇采用下面这条边界：

1. 保留当前 `models/qwen3.py` 作为 dense 主线，不直接改。
2. 新增 `models/qwen3_moe_proto.py` 作为教学原型。
3. 在这个原型文件里完整实现 Router、Top-k expert 选择、Expert 内部 FFN、输出加权合并。
4. 测试也写成独立实验测试，不污染现有主线测试。

---

## 4. 新增 `models/qwen3_moe_proto.py`

下面给出完整教学原型。这个文件不依赖不存在的外部模块，直接复用当前仓库已有的 `MergedLinear`、`RowLinear`、`SiluAndMul`、`RMSNorm`。

```python
"""教学版 Qwen3-MoE 原型。

这个文件的目标不是替换当前 dense Qwen3 主线，
而是用当前仓库已经有的算子风格，讲清楚单卡 MoE FFN 的执行语义。
"""

import torch
from torch import nn

from layers.activation import SiluAndMul
from layers.linear import MergedLinear, RowLinear
from layers.layernorm import RMSNorm


class MoERouter(nn.Module):
    """
    教学版 MoE Router。

    输入：
    - hidden_states: [num_tokens, hidden_size]

    输出：
    - topk_scores: [num_tokens, top_k]
    - topk_indices: [num_tokens, top_k]
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        assert num_experts > 0, "num_experts 必须 > 0"
        assert top_k > 0, "top_k 必须 > 0"
        assert top_k <= num_experts, "top_k 不能大于 num_experts"

        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_scores, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        return topk_scores, topk_indices


class MoEExpert(nn.Module):
    """
    单个 expert 的内部结构。

    它直接复用当前仓库 dense MLP 的设计：
    - gate_up_proj
    - SwiGLU
    - down_proj
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class Qwen3MoEMLP(nn.Module):
    """
    教学版单卡 MoE MLP。

    执行流程：
    1. Router 给每个 token 选 top-k experts。
    2. 每个 expert 只处理自己负责的 token 子集。
    3. 按 router score 把多个 expert 输出加权合并。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [MoEExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - hidden_states: [num_tokens, hidden_size]

        输出：
        - [num_tokens, hidden_size]
        """
        topk_scores, topk_indices = self.router(hidden_states)

        num_tokens = hidden_states.shape[0]
        output = torch.zeros_like(hidden_states)

        for token_idx in range(num_tokens):
            token_hidden = hidden_states[token_idx : token_idx + 1]
            token_output = torch.zeros_like(token_hidden)

            for expert_rank in range(self.top_k):
                expert_id = int(topk_indices[token_idx, expert_rank].item())
                expert_score = topk_scores[token_idx, expert_rank]

                expert_out = self.experts[expert_id](token_hidden)
                token_output = token_output + expert_score * expert_out

            output[token_idx : token_idx + 1] = token_output

        return output


class Qwen3MoEDecoderLayerProto(nn.Module):
    """
    教学版 MoE Decoder Layer 原型。

    这里只替换 FFN 为 MoE，Attention 主线保持接口形状兼容。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        attention_module: nn.Module,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = attention_module
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.moe_mlp = Qwen3MoEMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.moe_mlp(hidden_states)
        return hidden_states, residual
```

### 4.1 这份原型为什么不用高性能 dispatch

因为当前仓库还没有：

- Group-GEMM 基础设施。
- expert parallel worker 切分。
- all-to-all 路由通信。

所以这一篇的第一目标不是追求最快，而是先把 router 输出、token 到 expert 的映射、expert 计算、多 expert 输出合并讲清楚。

---

## 5. 为什么不直接改 `models/qwen3.py`

直接改主线会产生三个问题：

1. 当前 dense Qwen3 主线会被 MoE 逻辑污染，Day2 文档不再成立。
2. 当前 loader、权重映射、测试全都还是 dense 语义。
3. 读者会误以为“只要把 MLP 替换成 ModuleList experts，就等于支持了 Qwen3-MoE”。

所以这一篇必须明确：

> 当前仓库现在还没有真正接入 MoE 主线，这里只有一个完整可运行的教学原型，用来帮助你理解结构边界。

---

## 6. 新增 `tests/test_Day11_moe_proto.py`

这份测试只锁单卡 MoE 语义，不碰真实主线模型。

```python
"""Day11 MoE 原型结构测试。"""

import sys

sys.path.insert(0, ".")

import torch
from torch import nn

from models.qwen3_moe_proto import MoERouter, Qwen3MoEMLP, Qwen3MoEDecoderLayerProto


class FakeAttention(nn.Module):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_router_output_shape():
    router = MoERouter(hidden_size=16, num_experts=4, top_k=2)
    x = torch.randn(5, 16)
    scores, indices = router(x)

    assert scores.shape == (5, 2)
    assert indices.shape == (5, 2)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_moe_mlp_output_shape_matches_input_hidden_size():
    moe = Qwen3MoEMLP(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
    )
    x = torch.randn(6, 16)
    y = moe(x)
    assert y.shape == x.shape


@torch.inference_mode()
def test_moe_decoder_layer_proto_shape_contract():
    layer = Qwen3MoEDecoderLayerProto(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        attention_module=FakeAttention(),
    )

    positions = torch.arange(6)
    hidden_states = torch.randn(6, 16)
    out, residual = layer(positions, hidden_states)

    assert out.shape == hidden_states.shape
    assert residual.shape == hidden_states.shape
```

---

## 7. 如果以后真要把 MoE 接进当前主线

建议顺序是：

1. **先扩 `Config` 与模型注册语义。**
   明确当前加载的是 dense Qwen3 还是 Qwen3-MoE。
2. **再扩 loader / packed_modules_mapping。**
   让 expert 权重和 router 权重能被正确映射。
3. **再改 `models/qwen3.py` 的 DecoderLayer 组装逻辑。**
   在 dense MLP 与 MoE MLP 间分支。
4. **最后才碰 expert parallel、Group-GEMM 和量化。**
   这些都是性能层面的升级，不应先于语义层升级。

最不建议的顺序是：

> 先引入 Triton Group-GEMM，再回头想 router 和 token dispatch 到底对没对。

---

## 8. 验收命令

```bash
python -m py_compile models/qwen3_moe_proto.py tests/test_Day11_moe_proto.py
python tests/test_Day11_moe_proto.py
```

如果你想做轻量手测，还可以跑：

```bash
python - <<'PY'
import torch
from models.qwen3_moe_proto import Qwen3MoEMLP

moe = Qwen3MoEMLP(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2)
x = torch.randn(3, 8)
y = moe(x)
print("input shape:", x.shape)
print("output shape:", y.shape)
PY
```

---

## 9. 常见坑

1. **把 MoE 理解成“多个 MLP 求平均”。**
   真正的关键是 router 为每个 token 动态选 expert，而不是固定平均。
2. **为了讲 MoE，直接污染 dense 主线代码。**
   这会破坏当前 `00~07` 的教学闭环。
3. **一上来就把 expert parallel、量化、speculative decoding 全部混在一起。**
   这样没有任何一层边界是清楚的。
4. **把 router 输出当作 hard one-hot，不做 top-k 权重归一化。**
   教学版最稳妥的是 soft top-k combine，这样更容易看清 weighted merge 语义。
5. **以为有了 `ModuleList(experts)` 就等于已经支持了 Qwen3-MoE。**
   真正能不能支持，还取决于权重映射、配置语义、测试、并行与性能路径。

---

## 10. 本篇结束后你应该明白

这一篇最重要的不是“会写一个 router 层”。

真正要学会的是：

1. dense 到 MoE 的变化，核心发生在 FFN 路径，而不是 Attention 路径。
2. 单卡 MoE 的四个关键动作是：router、dispatch、expert compute、weighted combine。
3. 当前仓库最正确的做法，是先用独立教学原型把结构认知讲透，再决定是否改主线。
4. expert parallel、量化、Group-GEMM 都是后续性能层升级，不能代替语义正确性。

下一篇进入 FP8 与 KV cache 量化实验：

- `12-实现FP8与KV-Cache量化实验篇.md`
