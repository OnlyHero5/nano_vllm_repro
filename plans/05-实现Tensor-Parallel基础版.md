# 05. 实现 Tensor Parallel 基础版

这一篇开始做多卡，但只做教学版。

目标是：

> 在不改掉公开类名的前提下，把 Linear 和 Qwen3 Attention 升级成最小可用的 Tensor Parallel。

本篇不做 worker 池、不做 RPC、不做复杂调度。

本篇只做三件事：

1. Linear 层支持按 rank 切分权重。
2. Qwen3 模型区分全局 head 数和本地 head 数。
3. `ModelRunner` 能在 `torchrun` 环境下初始化分布式。

---

## 1. 前置条件

默认你已经完成：

1. `04` 单卡推理主循环。
2. `02` 模型主干和 `compute_logits()` 边界。

如果单卡链路还没稳定，先不要做 TP。

---

## 2. 当前代码是什么状态

当前 `layers/linear.py` 还是单卡版：

- `QKVLinear`
- `MergedLinear`
- `RowLinear`

它还没有：

- `rank / world_size` helper
- 按输出维切分的 Column Parallel
- 按输入维切分的 Row Parallel
- `all_reduce`

当前 `models/qwen3.py` 也还是单卡 head 语义：

```python
self.num_heads = num_heads
self.num_kv_heads = num_kv_heads
```

TP 后每张卡只拿一部分 head，所以模型里要同时知道：

- 全局 head 数
- 当前 rank 的本地 head 数

---

## 3. 修改 `layers/linear.py`

### 3.1 增加 TP helper

```python
import torch.distributed as dist
import torch.nn.functional as F


def divide(numerator: int, denominator: int) -> int:
    """
    整除切分。

    TP 里很多维度必须能被 world_size 整除。
    与其让 shape 在后面炸，不如这里直接报清楚。
    """
    assert denominator > 0, "denominator 必须 > 0"
    assert numerator % denominator == 0, f"{numerator} 不能被 {denominator} 整除"
    return numerator // denominator


def get_tp_world_size() -> int:
    """
    当前 TP world size。

    没有初始化 torch.distributed 时，退化成单卡。
    这样 Day1-Day5 的单卡测试不会被 TP 代码打碎。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tp_rank() -> int:
    """
    当前 rank。

    单进程运行时返回 0。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
```

### 3.2 写一个基础 Linear

```python
class LinearBase(nn.Module):
    """
    TP Linear 的公共基类。

    tp_dim 表示权重按哪个维度切：
    - 0：按输出维切，也就是 Column Parallel。
    - 1：按输入维切，也就是 Row Parallel。
    - None：不切。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
```

### 3.3 Column Parallel：按输出维切

```python
class ColumnParallelLinear(LinearBase):
    """
    按输出维切分的 Linear。

    原始权重形状：
        [global_output_size, input_size]

    每张卡保存：
        [global_output_size / tp_size, input_size]
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        local_output_size = divide(output_size, get_tp_world_size())
        super().__init__(input_size, local_output_size, bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # 当前 rank 只加载自己那一段输出维。
        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出是当前 rank 的局部输出，不需要 all_reduce。
        return F.linear(x, self.weight, self.bias)
```

### 3.4 Merged Column Parallel：用于 gate/up

```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    多个输出投影拼在一起的 Column Parallel。

    典型场景：
    - gate_proj
    - up_proj

    本地参数布局仍然是 [gate_local, up_local] 拼接。
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int,
    ) -> None:
        # shard_id=0 表示 gate，shard_id=1 表示 up。
        local_size = divide(self.output_sizes[shard_id], self.tp_size)
        local_offset = sum(self.output_sizes[:shard_id]) // self.tp_size

        # HF 权重是完整输出维，本 rank 只取自己的输出切片。
        shard = loaded_weight.chunk(self.tp_size, dim=self.tp_dim)[self.tp_rank]
        param.data[local_offset: local_offset + local_size].copy_(
            shard.to(device=param.device, dtype=param.dtype)
        )
```

### 3.5 QKV Parallel：用于 Q/K/V 融合投影

```python
class QKVParallelLinear(ColumnParallelLinear):
    """
    Q/K/V 融合后的 Column Parallel。

    全局布局：
        [Q_all, K_all, V_all]

    每张卡保存：
        [Q_local, K_local, V_local]
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
    ) -> None:
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        self.num_heads = divide(total_num_heads, get_tp_world_size())
        self.num_kv_heads = divide(total_num_kv_heads, get_tp_world_size())

        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_dim
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> None:
        """
        loaded_weight 是 HF 的 q_proj/k_proj/v_proj 之一。
        shard_id 告诉我们它该写入本地 qkv_proj 的哪一段。
        """
        assert shard_id in {"q", "k", "v"}

        if shard_id == "q":
            local_size = self.num_heads * self.head_dim
            local_offset = 0
        elif shard_id == "k":
            local_size = self.num_kv_heads * self.head_dim
            local_offset = self.num_heads * self.head_dim
        else:
            local_size = self.num_kv_heads * self.head_dim
            local_offset = self.num_heads * self.head_dim + self.num_kv_heads * self.head_dim

        shard = loaded_weight.chunk(self.tp_size, dim=self.tp_dim)[self.tp_rank]
        param.data[local_offset: local_offset + local_size].copy_(
            shard.to(device=param.device, dtype=param.dtype)
        )
```

### 3.6 Row Parallel：按输入维切，输出要规约

```python
class RowParallelLinear(LinearBase):
    """
    按输入维切分的 Linear。

    原始权重形状：
        [output_size, global_input_size]

    每张卡保存：
        [output_size, global_input_size / tp_size]

    每张卡先算局部输出，然后 all_reduce 求和。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        local_input_size = divide(input_size, get_tp_world_size())
        super().__init__(local_input_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.data.ndim == 1:
            # bias 不按输入维切，直接加载完整 bias。
            param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            return

        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 当前 rank 先算自己的局部输入贡献。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # 多卡时把所有 rank 的局部输出加起来。
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

### 3.7 保留公开类名

文件末尾保留当前仓库已经在用的名字：

```python
QKVLinear = QKVParallelLinear
MergedLinear = MergedColumnParallelLinear
RowLinear = RowParallelLinear
```

这样 `models/qwen3.py` 不需要到处改 import。

---

## 4. 修改 `Qwen3Attention`

模型层要区分“全局头数”和“本地头数”。

```python
def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int | None = None,
    max_position: int = 4096 * 32,
    rms_norm_eps: float = 1e-6,
    qkv_bias: bool = False,
    rope_theta: float = 1_000_000.0,
    layer_idx: int = 0,
) -> None:
    super().__init__()

    tp_size = get_tp_world_size()

    # 全局头数来自模型配置。
    self.total_num_heads = num_heads
    self.total_num_kv_heads = num_kv_heads

    # 本地头数是当前 rank 真正负责计算的头数。
    self.num_heads = divide(self.total_num_heads, tp_size)
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)

    self.head_dim = head_dim or hidden_size // self.total_num_heads

    # qkv_proj 返回的是本地 Q/K/V，所以切分尺寸也必须用本地头数。
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim ** -0.5

    self.qkv_proj = QKVLinear(
        hidden_size=hidden_size,
        head_dim=self.head_dim,
        total_num_heads=self.total_num_heads,
        total_num_kv_heads=self.total_num_kv_heads,
        bias=qkv_bias,
    )

    # RowLinear 的 input_size 传全局 q_size。
    # RowParallelLinear 内部会按 tp_size 切成本地输入维。
    self.o_proj = RowLinear(
        self.total_num_heads * self.head_dim,
        hidden_size,
        bias=False,
    )

    self.rotary_emb = get_rope(
        head_size=self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position,
        base=rope_theta,
    )

    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    self.attn = Attention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        scale=self.scaling,
        num_kv_heads=self.num_kv_heads,
        layer_idx=layer_idx,
    )
```

`forward()` 的主线不变，只是 `q_size / kv_size / num_heads / num_kv_heads` 已经变成本地值：

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q = q.view(num_tokens, self.num_heads, self.head_dim)
    k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
    v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)

    attn_output = self.attn(q, k, v)
    return self.o_proj(attn_output.reshape(num_tokens, -1))
```

---

## 5. 修改 `ModelRunner`

### 5.1 增加 TP 初始化

```python
def setup_tp_runtime(self) -> None:
    """
    初始化教学版 TP 运行时。

    tensor_parallel_size == 1:
        直接单卡运行。

    tensor_parallel_size > 1:
        需要用 torchrun 启动，并读取 RANK / LOCAL_RANK / WORLD_SIZE。
    """
    self.tp_size = self.config.tensor_parallel_size
    self.rank = 0
    self.local_rank = 0
    self.is_distributed = False

    if self.tp_size == 1:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Tensor Parallelism 需要 CUDA 环境")

    self.rank = int(os.environ["RANK"])
    self.local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == self.tp_size, (
        f"WORLD_SIZE={world_size} 与 tensor_parallel_size={self.tp_size} 不一致"
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.cuda.set_device(self.local_rank)
    self.device = torch.device("cuda", self.local_rank)
    self.is_distributed = True


def is_main_process(self) -> bool:
    return self.rank == 0
```

在 `__init__()` 开头调用：

```python
self.config = config
self.setup_tp_runtime()
```

### 5.2 KV Cache 用本地 KV 头数

多卡时每个 rank 只负责自己的 KV heads。

所以 `allocate_kv_cache()` 里不要继续用全局 KV 头数。

```python
self.num_kv_heads = self.model.config.num_key_value_heads // self.config.tensor_parallel_size
```

否则每张卡都会多分一份不属于自己的 KV Cache。

---

## 6. 新增 `tests/test_Day6_tp.py`

这份测试先锁两件事：

1. 单进程 fallback 能跑。
2. TP-aware 代码在 world_size=1 时不破坏旧行为。

```python
"""Day6 Tensor Parallel 基础测试。"""

import sys
sys.path.insert(0, ".")

import torch

from config import Config
from layers.linear import divide, get_tp_rank, get_tp_world_size


def test_tp_helpers_fallback():
    assert divide(8, 2) == 4
    assert get_tp_world_size() >= 1
    assert get_tp_rank() >= 0


def test_config_accepts_tp_size():
    config = Config(
        model_path="models/Qwen3-0.6B",
        tensor_parallel_size=1,
    )
    assert config.tensor_parallel_size == 1


@torch.inference_mode()
def test_qwen3_attention_single_rank_fallback():
    from models.qwen3 import Qwen3Attention

    attn = Qwen3Attention(
        hidden_size=128,
        num_heads=8,
        num_kv_heads=2,
        head_dim=16,
        qkv_bias=False,
    )

    positions = torch.arange(4)
    hidden_states = torch.randn(4, 128)
    output = attn(positions, hidden_states)

    assert output.shape == hidden_states.shape
```

如果机器有两张 GPU，再手动跑：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

---

## 7. 验收命令

```bash
python -m py_compile layers/linear.py models/qwen3.py engine/model_runner.py tests/test_Day6_tp.py
python tests/test_Day6_tp.py
```

有多卡环境时再跑：

```bash
torchrun --nproc_per_node=2 tests/test_Day6_tp.py
```

---

## 8. 常见坑

1. **没有分布式初始化就 import 失败**
   helper 必须在未初始化时返回 `world_size=1, rank=0`。

2. **QKV 切分还用全局 head 数**
   TP 后 `q_size / kv_size` 必须用本地 head 数。

3. **Row Parallel 忘记 all_reduce**
   每张卡只算局部输入贡献，不规约结果就不完整。

4. **KV Cache 继续按全局 KV heads 分配**
   会浪费显存，也会让后续 attention shape 对不上。

5. **直接改掉公开类名**
   保留 `QKVLinear / MergedLinear / RowLinear`，减少对其他文件的冲击。

---

## 9. 本篇结束后你应该明白

TP 最核心的是三条规则：

1. Column Parallel 按输出维切。
2. Row Parallel 按输入维切，结果要 all_reduce。
3. 模型层要区分全局 head 数和本地 head 数。

下一篇进入 CUDA Graph：

- `06-实现CUDA-Graph基础版.md`
