# 13. 实现 GPU Offload 与跨后端扩展总览

这一篇是进阶实验线的收口。

前面 `08~12` 讲的是：

- 怎么在当前主线边界上继续补调度、prefix cache、speculative decoding、MoE、低精度 KV cache

这一篇讲的是另一类能力：

> 当一张 GPU 不够、某个后端不合适，或者你想把同一套推理思想迁移到别的运行环境时，应该如何理解 `GPU offload`、`JAX / MLX` 和应用适配这几条路线。

这一篇只做四件事：

1. 讲清 `GPU offload` 在当前仓库里的真正接入点。
2. 新增一个完整可运行的 `gpu_offload_proto.py` 独立实验脚本。
3. 解释 `JAX / MLX / C++ / TTS 适配` 为什么更适合作为“方法迁移”，而不是“直接复制代码”。
4. 给出从当前仓库继续走向别的后端时，最应该保留的抽象边界。

这一篇**不做**下面这些事：

- 不把 GPU offload 真正并回当前主线。
- 不伪造 JAX / MLX 后端文件进入当前仓库主线。
- 不把 TTS 适配说成“当前 Qwen3 dense 主线已经支持语音模型”。
- 不在当前 `ModelRunner` 上硬插一堆异构设备分支。
- 不把“跨后端迁移”误讲成“只换几行 import”。

原因很简单：

> 这类能力真正可迁移的，不是单个 API，而是系统结构：`Sequence`、`BlockManager`、`Scheduler`、`ModelRunner`、`Context`、`Attention` 之间的职责边界。

---

## 1. 为什么 GPU offload 要单独讲

当前仓库默认假设：

- 模型权重主要放在 GPU。
- KV cache 也放在 GPU。
- Decode attention 直接从 GPU cache 里读。

这在显存足够时很好，但当下面几种情况出现时就不够了：

1. 模型权重太大，单卡放不下。
2. KV cache 在长上下文 / 高并发下占用太高。
3. 你想在“有限 GPU + 更多 CPU 内存”的环境里继续跑系统实验。

这时候 `GPU offload` 的核心问题不是“能不能搬数据”，而是：

> 哪些数据要常驻 GPU，哪些数据可以冷下来之后再搬回 CPU。

在当前教学仓库的语义里，这会自然拆成两条线：

### 1.1 权重 offload

- 权重平时放 CPU 或 host memory。
- 真正用到某层时再搬到 GPU。

### 1.2 KV cache offload

- 热的 block 留在 GPU。
- 冷的 block 搬回 CPU。
- 需要再次访问时再换回 GPU。

其中，KV cache offload 更贴近当前 `BlockManager` 主线，因为：

- 它本来就已经按 block 管理 cache。
- “热块 / 冷块”的概念天然和 block 资源管理一致。

---

## 2. 当前代码是什么状态

与 offload 最相关的当前文件有：

1. `engine/model_runner.py`
2. `engine/block_manager.py`
3. `utils/context.py`
4. `layers/attention.py`

### 2.1 当前 `ModelRunner` 假设模型已经整体放到 `self.device`

当前 `_load_model()` 的关键逻辑是：

```python
model = model.to(self.device, dtype=torch.bfloat16)
model.eval()
```

这意味着当前主线没有：

- 分层按需上卡。
- 参数页级换入换出。
- CPU / GPU 权重双副本管理。

### 2.2 当前 `kv_cache` 是一整套 GPU tensor 列表

当前结构是：

```python
self.kv_cache: list[torch.Tensor]
```

每层一个：

```python
[2, num_blocks, block_size, num_kv_heads, head_dim]
```

这意味着当前主线没有：

- block 级“驻留在 CPU 还是 GPU”的状态。
- 冷热块迁移策略。
- 恢复路径。

### 2.3 当前 `BlockManager` 天然适合作为 KV offload 的入口

这是这一篇最重要的观察。

因为 `BlockManager` 已经知道：

- 哪些 block 在用。
- 哪些 block 空闲。
- 哪些 block 被哪条序列引用。
- 哪些 block 命中过 prefix cache。

所以如果以后真要做 KV cache offload，最合理的入口就是：

> 继续让 `BlockManager` 管“一个 block 逻辑上归谁”，再额外引入“这个 block 物理上现在驻留在哪个设备”。

---

## 3. 当前教学仓库里最适合的实验边界

这一篇采用下面这条边界：

1. **不改主线 `ModelRunner` 和 `Attention`。**
2. **新增独立实验文件 `utils/gpu_offload_proto.py`。**
3. **在原型里完整实现：**
   - block 级驻留状态。
   - GPU <-> CPU 迁移。
   - LRU 风格热块淘汰。
   - 按 block 读取前自动换入。
4. **测试也写成独立实验测试。**

这样做有两个好处：

- 你能真正把“offload 的系统语义”跑起来。
- 不会误导读者以为当前主线已经天然支持异构内存推理。

---

## 4. 新增 `utils/gpu_offload_proto.py`

下面给出完整教学原型。这个原型不依赖当前仓库不存在的自定义 runtime，重点是把“block 现在住在哪里”讲清楚。

```python
"""教学版 GPU offload 原型。

目标：
1. 按 block 管理张量驻留位置。
2. 当 GPU 容量不足时，把冷 block 换出到 CPU。
3. 再次访问该 block 时，再自动换回 GPU。

注意：
- 这不是当前主线的真实 KV cache 实现。
- 它是一个独立实验，用于说明 offload 的最小系统语义。
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class OffloadBlock:
    """一个可换入换出的 block。"""

    cpu_tensor: torch.Tensor
    gpu_tensor: torch.Tensor | None
    resident: str


class GPUOffloadCacheProto:
    """教学版 block 级 offload cache。"""

    def __init__(
        self,
        num_blocks: int,
        block_shape: tuple[int, ...],
        max_gpu_blocks: int,
        dtype=torch.float16,
    ):
        assert num_blocks > 0, "num_blocks 必须 > 0"
        assert max_gpu_blocks > 0, "max_gpu_blocks 必须 > 0"
        assert max_gpu_blocks <= num_blocks, "max_gpu_blocks 不能大于 num_blocks"

        self.num_blocks = num_blocks
        self.block_shape = block_shape
        self.max_gpu_blocks = max_gpu_blocks
        self.dtype = dtype
        self.gpu_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.blocks: list[OffloadBlock] = []
        for _ in range(num_blocks):
            cpu_tensor = torch.zeros(block_shape, dtype=dtype, device="cpu")
            self.blocks.append(OffloadBlock(cpu_tensor=cpu_tensor, gpu_tensor=None, resident="cpu"))

        self.gpu_lru: OrderedDict[int, None] = OrderedDict()

    def _touch_lru(self, block_id: int) -> None:
        if block_id in self.gpu_lru:
            self.gpu_lru.move_to_end(block_id)
        else:
            self.gpu_lru[block_id] = None

    def _evict_one_block(self) -> None:
        assert len(self.gpu_lru) > 0, "没有可淘汰的 GPU block"
        evict_block_id, _ = self.gpu_lru.popitem(last=False)
        block = self.blocks[evict_block_id]

        if block.gpu_tensor is not None:
            block.cpu_tensor.copy_(block.gpu_tensor.to("cpu"))
            block.gpu_tensor = None
            block.resident = "cpu"

    def ensure_on_gpu(self, block_id: int) -> torch.Tensor:
        assert 0 <= block_id < self.num_blocks, "block_id 越界"
        block = self.blocks[block_id]

        if self.gpu_device.type == "cpu":
            return block.cpu_tensor

        if block.resident == "gpu" and block.gpu_tensor is not None:
            self._touch_lru(block_id)
            return block.gpu_tensor

        if len(self.gpu_lru) >= self.max_gpu_blocks:
            self._evict_one_block()

        block.gpu_tensor = block.cpu_tensor.to(self.gpu_device)
        block.resident = "gpu"
        self._touch_lru(block_id)
        return block.gpu_tensor

    def write_block(self, block_id: int, data: torch.Tensor) -> None:
        assert data.shape == self.block_shape, f"写入形状不匹配: {data.shape} vs {self.block_shape}"
        block = self.blocks[block_id]
        block.cpu_tensor.copy_(data.to("cpu", dtype=self.dtype))

        if block.gpu_tensor is not None:
            block.gpu_tensor.copy_(data.to(self.gpu_device, dtype=self.dtype))
            self._touch_lru(block_id)

    def read_block(self, block_id: int, prefer_gpu: bool = True) -> torch.Tensor:
        if prefer_gpu:
            return self.ensure_on_gpu(block_id)
        return self.blocks[block_id].cpu_tensor

    def get_residency_report(self) -> dict:
        gpu_blocks = sum(1 for block in self.blocks if block.resident == "gpu")
        cpu_blocks = self.num_blocks - gpu_blocks
        return {
            "num_blocks": self.num_blocks,
            "gpu_blocks": gpu_blocks,
            "cpu_blocks": cpu_blocks,
            "max_gpu_blocks": self.max_gpu_blocks,
        }
```

### 4.1 这份原型真正解释了什么

它不是在证明“当前仓库已经能 offload”。

它真正解释的是：

1. **block 级驻留状态**
   - 一个 block 逻辑上存在，但物理上可以住在 CPU 或 GPU。
2. **冷块淘汰**
   - GPU 装不下时，要有明确的换出策略。
3. **再次访问时的自动换入**
   - Decode attention 真正关心的是“当前要读的块能不能马上用”。

这三件事，才是 offload 的系统核心。

---

## 5. 新增 `tests/test_Day13_gpu_offload_proto.py`

这份测试不依赖当前主线模型，只验证 offload 的驻留语义。只有当你把第 4 节代码保存为 `utils/gpu_offload_proto.py` 后，下面的测试才可以直接运行。

```python
"""Day13 GPU offload 原型测试。"""

import sys

import torch

sys.path.insert(0, ".")

from utils.gpu_offload_proto import GPUOffloadCacheProto


def test_initial_residency_is_cpu():
    cache = GPUOffloadCacheProto(
        num_blocks=4,
        block_shape=(2, 3),
        max_gpu_blocks=2,
    )
    report = cache.get_residency_report()

    assert report["num_blocks"] == 4
    assert report["cpu_blocks"] >= 2


def test_write_then_read_block_keeps_shape():
    cache = GPUOffloadCacheProto(
        num_blocks=4,
        block_shape=(2, 3),
        max_gpu_blocks=2,
    )
    x = torch.randn(2, 3)
    cache.write_block(0, x)
    y = cache.read_block(0, prefer_gpu=False)

    assert y.shape == x.shape
    assert torch.allclose(y.cpu(), x.cpu().to(y.dtype), atol=1e-5, rtol=1e-5)


def test_gpu_capacity_is_bounded_even_when_touching_many_blocks():
    cache = GPUOffloadCacheProto(
        num_blocks=4,
        block_shape=(2, 3),
        max_gpu_blocks=2,
    )

    for i in range(4):
        x = torch.full((2, 3), float(i))
        cache.write_block(i, x)
        _ = cache.read_block(i, prefer_gpu=True)

    report = cache.get_residency_report()
    assert report["gpu_blocks"] <= report["max_gpu_blocks"]
```

---

## 6. 如果以后真要接回当前主线，最自然的边界在哪

这一篇不直接改主线，但必须把未来路径讲清楚。

建议按下面顺序走：

### 6.1 先扩 `BlockManager` 的 block 元数据

当前 `Block` 里只有：

- `block_id`
- `ref_count`
- `hash`
- `token_ids`

如果以后做真实 KV cache offload，建议再加：

- `resident_device`
- `last_access_step`
- `pinned`

这样 `BlockManager` 就不只知道“这个 block 归谁”，还知道“这个 block 现在住在哪”。

### 6.2 再扩 `ModelRunner` 的 KV cache 分配语义

当前 `allocate_kv_cache()` 假设：

- 所有层的所有块都在 GPU 上直接分好。

真实 offload 路径下，应该拆成：

- GPU hot cache。
- CPU backing store。

也就是说，`ModelRunner` 要负责：

- 分配热层缓存。
- 分配 CPU 持久副本。
- 给 `Attention` 提供当前可读的热块视图。

### 6.3 最后才碰 `layers/attention.py`

因为 `Attention` 真正关心的是：

- 这轮 decode 要读哪些 block。
- 这些 block 当前是否已经在 GPU。

换句话说：

> Offload 不应该先从 `Attention` 开始改，而应该先从“block 驻留与换入换出策略”开始改。

---

## 7. JAX / MLX / C++ / TTS 适配该怎么看

这一节不写实现代码，因为重点不是“把这个仓库原地变成 JAX 或 MLX 项目”，而是说明：

> 当你迁移到别的后端时，真正应该带走的是系统结构，不是逐行 API。

### 7.1 JAX 路线

JAX / Pallas 路线最值得带走的，不是 `torch.Tensor -> jax.Array` 这种表层变换。

真正该带走的是：

- `Sequence` 的运行时账本语义。
- `Scheduler` 的 token budget 语义。
- `BlockManager` 的 block / prefix cache 语义。
- `Context` 的“把 attention 元数据从调度层传到算子层”的设计。

也就是说：

> 后端变了，但“请求状态 / block 资源 / 调度 / 算子元数据”的职责拆分不应该变。

### 7.2 MLX 路线

MLX 方向最值得吸收的是：

- Apple 统一内存让“热 / 冷块”和“显式 offload”的边界变得不同。
- 但 `block`、`prefix cache`、`chunked prefill` 这些设计仍然成立。

也就是说：

> 内存体系变了，不代表调度和 cache 结构就失效了。

### 7.3 C++ / CUDA 路线

C++ / CUDA 的价值主要在于：

- 把 Python 控制流明确压缩成更底层的 runtime 组件。
- 把 PagedAttention、block 管理、decode kernel 做成更强性能实现。

但迁移时最该保留的，仍然是：

- block table 语义。
- prefill / decode 分离。
- request scheduling 语义。

### 7.4 TTS / 应用适配路线

像 VoxCPM 这类 TTS 适配，最值得学的是：

- 推理系统并不只服务文本 LLM。
- continuous batching、KV cache、spec decode、prefix reuse 这些思想，依然可以迁移到别的生成模型。

但不能误解成：

> 当前这个 Qwen3 dense 教学仓，已经天然等价于一个 TTS 服务框架。

它只是说明：

- 你现在学会的这些系统设计，未来可以迁移到别的模型类型。

---

## 8. 验收命令

下面命令只有在你已经把第 4 节和第 5 节代码分别保存到 `utils/gpu_offload_proto.py` 与 `tests/test_Day13_gpu_offload_proto.py` 后才可运行：

```bash
python -m py_compile utils/gpu_offload_proto.py tests/test_Day13_gpu_offload_proto.py
python tests/test_Day13_gpu_offload_proto.py
```

如果你只想快速看驻留状态变化，也可以把第 4 节原型保存成文件后运行：

```bash
python - <<'PY'
import torch

from utils.gpu_offload_proto import GPUOffloadCacheProto

cache = GPUOffloadCacheProto(num_blocks=4, block_shape=(2, 3), max_gpu_blocks=2)
for i in range(4):
    x = torch.full((2, 3), float(i))
    cache.write_block(i, x)
    _ = cache.read_block(i, prefer_gpu=True)
    print(i, cache.get_residency_report())
PY
```

你应该能看到：

- 随着访问更多 block，GPU 驻留块数不会无限增长。
- 系统会持续保持“最多只有 `max_gpu_blocks` 个热块”。

---

## 9. 常见坑

1. **把 offload 理解成“读不下就 `to('cpu')`”。**

   真正的关键是：

   - 哪些块常驻。
   - 哪些块淘汰。
   - 什么时候再换入。

2. **一上来就修改 `Attention` 主线。**

   当前教学仓库最自然的切入点是 block 驻留状态，而不是先改 kernel。

3. **把 JAX / MLX 迁移理解成“换语言、换框架”，却忘了保留系统结构。**

   真正可迁移的是职责边界，不只是 API 名字。

4. **把 TTS 适配误讲成“当前文本主线已经支持语音模型”。**

   正确说法应该是：当前推理系统思想可迁移，但模型语义、输入输出和调度细节仍然不同。

5. **把原型代码写成主线既有能力。**

   这一篇所有代码都必须明确是“独立实验原型”，不是当前主线已合并功能。

---

## 10. 本篇结束后你应该明白

这一篇最重要的不是“会写一个 LRU 容器”。

真正要学会的是：

1. `GPU offload` 的本质是“块级驻留管理”，不是简单的数据搬运。
2. 当前仓库里，最自然的接入点是 `BlockManager + ModelRunner`，而不是直接从 `Attention` kernel 开始改。
3. JAX / MLX / C++ / TTS 迁移时，最值得保留的是系统结构和职责边界。
4. 当后端和模型类型变化时，真正通用的仍然是：
   - `Sequence` 管请求状态。
   - `BlockManager` 管块资源。
   - `Scheduler` 管 token budget 和执行顺序。
   - `ModelRunner / Context / Attention` 管“这一轮如何真正执行”。

到这里，进阶实验线 `08~13` 就收口了。下一步不是继续堆功能，而是回头对照当前仓库代码，判断：

- 哪些进阶能力已经值得真正并回主线。
- 哪些还应该先作为独立实验继续验证。
