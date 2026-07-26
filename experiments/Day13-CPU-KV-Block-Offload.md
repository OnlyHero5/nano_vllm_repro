# Day 13 — CPU KV Block Offload：显存不够时把 KV 换到内存

> **本篇边界**：这里落地的是 KV block 的 GPU↔CPU 换入换出（swap）——不是权重 offload，不涉及跨后端（JAX / MLX / C++）实现，也不含生产级 KV connector / LMCache / 异步 prefetch。一条 swap 通路讲透，比铺开三条讲不透更有用。
>
> **前置依赖**：本篇修改 `engine/sequence.py / engine/block_manager.py / engine/model_runner.py / engine/scheduler.py`，以主线 Day1-Day6 落地后的代码为基础。

当前主线在 GPU KV block 不够时，做法很粗暴：把 sequence preempt 回 `WAITING`，释放 KV cache，下一轮重新 prefill。已经算过的 KV 全扔了，重来一遍。

更好的做法是 **swap**：把 KV block 从 GPU copy 到 CPU 内存，腾出 GPU 块给别人用；等这条序列需要继续 decode 时，再从 CPU copy 回来。像操作系统的页面换入换出。

这次加一条**可选**的 CPU KV block offload 路径。默认关闭，主线行为不变；只有显式启用 `BlockManager(enable_cpu_kv_offload=True)` 并注册 KV copy handler 时才进入 offload 分支。

五个改动点：

1. `SequenceStatus.SWAPPED`：让请求状态能表达“KV 已换出到 CPU，等待换回 GPU”。
2. `BlockResidency` 元数据：让 `BlockManager` 同时知道 block 的逻辑归属和物理驻留位置。
3. `ModelRunner` CPU KV buffer：按层把 KV block 在 GPU cache 和 CPU buffer 之间 copy。
4. `Scheduler` 队列变化：GPU blocks 不够时换出 running sequence；需要继续 decode 时换回。
5. `tests/test_Day13_kv_offload.py`：不依赖真实大模型，只验证状态、元数据、copy 语义和队列状态。

生产级 KV connector / LMCache / async prefetch / JAX / MLX / C++ 后端都不做，也不做权重 offload。

---

## 1. 状态流

当前主线在 KV block 不够时，会把 sequence preempt 回 `WAITING`，释放 KV cache，下一轮重新 prefill。offload 路径改成：

```text
WAITING
  -> RUNNING     # prefill 分配 GPU KV blocks
  -> RUNNING     # decode 持续追加 token
  -> SWAPPED     # GPU KV blocks 不够，KV block copy 到 CPU buffer，GPU blocks 释放
  -> RUNNING     # 继续 decode 前，CPU KV blocks copy 回 GPU cache，重建 block_table
  -> FINISHED    # 请求完成，释放 GPU/CPU 侧元数据
```

进入 `SWAPPED` 的条件：

- sequence 已经在 `running` 队列中；
- `Scheduler` 发现当前 GPU KV blocks 不够；
- `BlockManager.can_swap_out(seq)` 返回 `True`；
- `BlockManager.swap_out(seq)` 完成每层 KV block 的 GPU -> CPU copy，并释放原 GPU block id。

退出 `SWAPPED` 的条件：

- sequence 位于 `swapped` 队列头部；
- `BlockManager.can_swap_in(seq)` 返回 `True`；
- `BlockManager.swap_in(seq)` 为它重新分配 GPU block id，完成每层 KV block 的 CPU -> GPU copy；
- `Scheduler` 把它放回 `running` 队列。

这条路径默认关闭，因此现有 `example.py`、`LLMEngine` 默认初始化、Day1~Day4 测试语义保持不变。

---

## 2. 文件级修改总览

按下面文件边界写补丁：

|文件 | 修改目的 |
| --- | --- |
| `engine/sequence.py` | 给 `SequenceStatus` 新增 `SWAPPED`，说明状态进入/退出语义 |
| `engine/block_manager.py` | 新增 `BlockResidency` 元数据、CPU/GPU 驻留状态、换出候选选择、`swap_out` / `swap_in` |
| `engine/model_runner.py` | 新增 CPU KV buffer；提供每层 KV block 的 GPU→CPU与 CPU→GPU copy helper |
| `engine/scheduler.py` | 新增 `swapped` 队列；GPU blocks 不够时换出 running seq；decode 前尝试换回 |
| `tests/test_Day13_kv_offload.py` | 完整测试文件，不加载真实模型权重 |

不需要改 `README.md`、`example.py`、`engine/llm_engine.py`、`layers/attention.py` 或任何已有 Day1-Day4 测试。

---

## 3. `engine/sequence.py`：新增 `SWAPPED` 状态

把 `SequenceStatus` 替换为下面版本，并给 `Sequence` 增加 `is_swapped` 属性。`SWAPPED` 表示：sequence 的 token 账本还在，但它的 GPU KV blocks 已经被释放，KV 内容驻留在 CPU buffer 中。

```python
class SequenceStatus(Enum):
    """
    序列状态枚举。

    WAITING: 尚未 prefill，等待分配 GPU KV blocks。
    RUNNING: KV blocks 驻留在 GPU，可以执行 prefill 或 decode。
    SWAPPED: KV blocks 已换出到 CPU，暂时不能直接 decode。
    FINISHED: 请求结束，GPU/CPU KV 资源都应释放。
    """
    WAITING = auto()
    RUNNING = auto()
    SWAPPED = auto()
    FINISHED = auto()
```

在 `Sequence` 的状态查询区加入：

```python
@property
def is_swapped(self):
    """检查 KV blocks 是否已换出到 CPU。"""
    return self.status == SequenceStatus.SWAPPED
```

状态约定：

- `Scheduler.add_sequence()` 仍然把新请求置为 `WAITING`。
- `Scheduler.schedule()` 成功 prefill 时置为 `RUNNING`。
- `Scheduler.swap_out_sequence()` 成功换出时置为 `SWAPPED`。
- `Scheduler.try_swap_in()` 成功换回时置为 `RUNNING`。
- `Scheduler.postprocess()` 命中 EOS 或 `max_tokens` 时置为 `FINISHED`。

---

## 4. `engine/block_manager.py`：block residency 元数据与 swap-in/out

### 4.1 扩展 import

在文件顶部扩展 import：

```python
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from engine.sequence import Sequence
```

### 4.2 新增 block residency 类型

把下面代码放在 `Block` 类之前：

```python
class BlockResidency(Enum):
    """KV block 当前物理驻留位置。"""

    FREE = "free"
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class BlockResidencyInfo:
    """一条 sequence 里的一个 logical block 的驻留账本。"""

    residency: BlockResidency = BlockResidency.FREE
    owner_seq_id: int | None = None
    logical_block_idx: int | None = None
    gpu_block_id: int | None = None
    cpu_slot_id: int | None = None


CopyGpuBlockToCpu = Callable[[int, int], None]
CopyCpuBlockToGpu = Callable[[int, int], None]
```

这里刻意区分两件事：

- `gpu_block_id`：当前 GPU KV cache tensor 中的物理 block id，也是 `seq.block_table` 在 `RUNNING` 时使用的 id。
- `cpu_slot_id`：CPU KV buffer 中的备份槽位。sequence 进入 `SWAPPED` 后，`seq.block_table` 会清空，逻辑 block 位置由 `BlockResidencyInfo(logical_block_idx=...)` 保留。

### 4.3 扩展 `Block`

在 `Block.__init__()` 中增加 `residency` 字段：

```python
self.residency = BlockResidency.FREE
```

把 `Block.reset()` 调整为：

```python
def reset(self):
    """重置 block，重新分配到 GPU 时调用。"""
    self.ref_count = 1
    self.hash = -1
    self.token_ids = []
    self.residency = BlockResidency.GPU
```

在 `Block.__repr__()` 中保留原有信息即可；为了调试也可以加入 `residency`：

```python
def __repr__(self):
    return (
        f"Block(id={self.block_id}, ref={self.ref_count}, "
        f"hash={self.hash}, residency={self.residency.value})"
    )
```

### 4.4 扩展 `BlockManager.__init__`

把 `BlockManager.__init__` 替换为下面版本。`enable_cpu_kv_offload=False` 是保持默认主线行为不变的关键。

```python
def __init__(
    self,
    num_blocks: int,
    block_size: int,
    enable_cpu_kv_offload: bool = False,
    num_cpu_blocks: int | None = None,
):
    """
    Args:
        num_blocks: GPU KV cache block 数。
        block_size: 每个 block 存储的 token 数。
        enable_cpu_kv_offload: 是否启用 Day13 教学版 CPU KV block offload。
        num_cpu_blocks: CPU KV buffer slot 数；默认与 GPU block 数一致。
    """
    self.block_size = block_size
    self.num_blocks = num_blocks

    self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
    self.free_block_ids: deque[int] = deque(range(num_blocks))
    self.used_block_ids: set[int] = set()
    self.hash_to_block_id: dict[int, int] = {}

    self.enable_cpu_kv_offload = enable_cpu_kv_offload
    self.num_cpu_blocks = num_cpu_blocks if num_cpu_blocks is not None else num_blocks
    self.free_cpu_slot_ids: deque[int] = deque(range(self.num_cpu_blocks))

    # key: (seq_id, logical_block_idx)
    self.block_residency: dict[tuple[int, int], BlockResidencyInfo] = {}

    # key: gpu_block_id, value: (seq_id, logical_block_idx)
    # Day13 教学路径只把 ref_count == 1 的 block 作为换出候选，因此这里不处理共享 block 的多 owner。
    self.gpu_block_owners: dict[int, tuple[int, int]] = {}

    self._copy_gpu_block_to_cpu: CopyGpuBlockToCpu | None = None
    self._copy_cpu_block_to_gpu: CopyCpuBlockToGpu | None = None
```

### 4.5 注册 `ModelRunner` copy handler

`BlockManager` 不直接持有 KV tensor，它只决定哪些 block 要搬迁。真正的 tensor copy 交给 `ModelRunner` 注册进来的函数。

```python
def register_kv_swap_handlers(
    self,
    copy_gpu_block_to_cpu: CopyGpuBlockToCpu,
    copy_cpu_block_to_gpu: CopyCpuBlockToGpu,
) -> None:
    """注册 KV block copy 函数，由 ModelRunner 提供真实 tensor copy。"""
    self._copy_gpu_block_to_cpu = copy_gpu_block_to_cpu
    self._copy_cpu_block_to_gpu = copy_cpu_block_to_gpu


def _require_kv_swap_handlers(self) -> None:
    if not self.enable_cpu_kv_offload:
        raise ValueError("CPU KV offload is disabled")
    if self._copy_gpu_block_to_cpu is None or self._copy_cpu_block_to_gpu is None:
        raise ValueError("KV swap handlers are not registered")
```

### 4.6 记录 GPU residency

在 `allocate()` 的 cache hit 和 cache miss 两个分支里，`seq.block_table.append(block_id)` 之后都调用 `_mark_gpu_resident(seq, i, block_id)`。

在 `append_slot()` 分配新 block 后，也调用 `_mark_gpu_resident(seq, logical_block_idx, new_block.block_id)`。

新增 helper：

```python
def _mark_gpu_resident(
    self,
    seq: Sequence,
    logical_block_idx: int,
    gpu_block_id: int,
) -> None:
    """记录 sequence 的 logical block 当前驻留在某个 GPU block。"""
    key = (seq.seq_id, logical_block_idx)
    old_info = self.block_residency.get(key)
    cpu_slot_id = old_info.cpu_slot_id if old_info is not None else None

    self.block_residency[key] = BlockResidencyInfo(
        residency=BlockResidency.GPU,
        owner_seq_id=seq.seq_id,
        logical_block_idx=logical_block_idx,
        gpu_block_id=gpu_block_id,
        cpu_slot_id=cpu_slot_id,
    )
    self.gpu_block_owners[gpu_block_id] = key
    self.blocks[gpu_block_id].residency = BlockResidency.GPU


def get_residency(self, seq: Sequence, logical_block_idx: int) -> BlockResidencyInfo:
    """读取某条 sequence 的某个 logical block 驻留信息。"""
    return self.block_residency[(seq.seq_id, logical_block_idx)]


def get_num_free_cpu_blocks(self) -> int:
    """CPU KV buffer 剩余 slot 数。"""
    return len(self.free_cpu_slot_ids)
```

`allocate()` 中两处应形成下面形态：

```python
seq.block_table.append(cached_block_id)
self._mark_gpu_resident(seq, i, cached_block_id)
```

```python
seq.block_table.append(block_id)
self._mark_gpu_resident(seq, i, block_id)
```

`append_slot()` 的新 block 分支应形成下面形态：

```python
new_block = self._allocate_fresh_block()
block_table.append(new_block.block_id)
logical_block_idx = len(block_table) - 1
self._mark_gpu_resident(seq, logical_block_idx, new_block.block_id)
```

### 4.7 更新释放逻辑

把 `_deallocate_block()` 调整为只释放 GPU block，不删除 prefix cache 的 hash 内容。这样原有 prefix cache 教学语义仍然保留。

```python
def _deallocate_block(self, block_id: int):
    """释放指定 GPU block。"""
    block = self.blocks[block_id]
    assert block.ref_count == 0, f"Block {block_id} still has references"
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
    block.residency = BlockResidency.FREE
```

新增 residency 清理 helper：

```python
def _delete_residency(self, seq: Sequence, logical_block_idx: int) -> None:
    key = (seq.seq_id, logical_block_idx)
    info = self.block_residency.pop(key, None)
    if info is None:
        return
    if info.gpu_block_id is not None:
        self.gpu_block_owners.pop(info.gpu_block_id, None)
    if info.cpu_slot_id is not None:
        self.free_cpu_slot_ids.append(info.cpu_slot_id)


def _release_cpu_residency_for_sequence(self, seq: Sequence) -> None:
    keys = [key for key in self.block_residency if key[0] == seq.seq_id]
    for _, logical_block_idx in keys:
        self._delete_residency(seq, logical_block_idx)
```

把 `deallocate()` 替换为下面版本，让 `FINISHED` sequence 能同时释放 GPU 与 CPU 侧元数据。

```python
def deallocate(self, seq: Sequence):
    """释放序列占用的 GPU blocks 和 CPU offload slots。"""
    for logical_block_idx, block_id in reversed(list(enumerate(seq.block_table))):
        block = self.blocks[block_id]
        block.ref_count -= 1

        if block.ref_count == 0:
            self._deallocate_block(block_id)

        self._delete_residency(seq, logical_block_idx)

    self._release_cpu_residency_for_sequence(seq)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

### 4.8 选择换出候选

教学补丁只选择 `ref_count == 1` 的 running sequence 作为换出候选。这样可以避开 prefix cache/shared block 的多 owner 复杂性，同时保留当前学习项目的重点：block 驻留状态与 copy 语义。

```python
def can_swap_out(self, seq: Sequence) -> bool:
    """检查 sequence 是否可以从 GPU KV cache 换出到 CPU KV buffer。"""
    if not self.enable_cpu_kv_offload:
        return False
    if not seq.block_table:
        return False
    if len(self.free_cpu_slot_ids) < len(seq.block_table):
        return False
    return all(self.blocks[block_id].ref_count == 1 for block_id in seq.block_table)


def select_swap_out_victim(self, candidates) -> Sequence | None:
    """从 running 队列尾部选择一个可换出的 sequence。"""
    for seq in reversed(list(candidates)):
        if self.can_swap_out(seq):
            return seq
    return None
```

### 4.9 `swap_out`：GPU KV block copy 到 CPU buffer，并释放 GPU blocks

```python
def swap_out(self, seq: Sequence) -> list[int]:
    """
    将 sequence 的所有 GPU KV blocks 换出到 CPU buffer。

    Returns:
        old_block_table: 换出前的 GPU block id 列表，用于测试和日志。
    """
    self._require_kv_swap_handlers()
    if not self.can_swap_out(seq):
        raise ValueError(f"Sequence {seq.seq_id} cannot be swapped out")

    old_block_table = seq.block_table.copy()

    for logical_block_idx, gpu_block_id in enumerate(old_block_table):
        cpu_slot_id = self.free_cpu_slot_ids.popleft()

        # 真正的数据搬运由 ModelRunner 完成：每一层的 KV block 都从 GPU copy 到 CPU。
        self._copy_gpu_block_to_cpu(gpu_block_id, cpu_slot_id)

        key = (seq.seq_id, logical_block_idx)
        info = self.block_residency.get(
            key,
            BlockResidencyInfo(
                owner_seq_id=seq.seq_id,
                logical_block_idx=logical_block_idx,
            ),
        )
        info.residency = BlockResidency.CPU
        info.gpu_block_id = None
        info.cpu_slot_id = cpu_slot_id
        self.block_residency[key] = info

        self.gpu_block_owners.pop(gpu_block_id, None)

        block = self.blocks[gpu_block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == gpu_block_id:
            del self.hash_to_block_id[block.hash]
        block.ref_count = 0
        block.hash = -1
        block.token_ids = []
        block.residency = BlockResidency.FREE

        if gpu_block_id in self.used_block_ids:
            self.used_block_ids.remove(gpu_block_id)
        if gpu_block_id not in self.free_block_ids:
            self.free_block_ids.append(gpu_block_id)

    seq.block_table.clear()
    return old_block_table
```

换出后，sequence 的 token 列表仍然完整；只是 `block_table` 清空，表示它当前没有可供 attention 读取的 GPU block table。

### 4.10 `swap_in`：CPU KV block copy 回 GPU cache，并重建 `block_table`

```python
def _get_sequence_residency_infos(self, seq: Sequence) -> list[BlockResidencyInfo]:
    infos = [
        info
        for (seq_id, _), info in self.block_residency.items()
        if seq_id == seq.seq_id
    ]
    infos.sort(key=lambda info: info.logical_block_idx)
    return infos


def can_swap_in(self, seq: Sequence) -> bool:
    """检查 sequence 是否可以从 CPU KV buffer 换回 GPU KV cache。"""
    if not self.enable_cpu_kv_offload:
        return False
    infos = self._get_sequence_residency_infos(seq)
    if not infos:
        return False
    if any(info.residency != BlockResidency.CPU for info in infos):
        return False
    return len(self.free_block_ids) >= len(infos)


def swap_in(self, seq: Sequence) -> list[int]:
    """
    将 sequence 的 CPU KV blocks 换回 GPU cache，并重建 seq.block_table。

    Returns:
        new_block_table: 换回后的 GPU block id 列表。
    """
    self._require_kv_swap_handlers()
    if not self.can_swap_in(seq):
        raise ValueError(f"Sequence {seq.seq_id} cannot be swapped in")

    infos = self._get_sequence_residency_infos(seq)
    new_block_table: list[int] = []
    prefix_hash = -1

    for info in infos:
        assert info.logical_block_idx is not None
        assert info.cpu_slot_id is not None

        block = self._allocate_fresh_block()
        gpu_block_id = block.block_id

        # 真正的数据搬运由 ModelRunner 完成：每一层的 KV block 都从 CPU copy 回 GPU。
        self._copy_cpu_block_to_gpu(info.cpu_slot_id, gpu_block_id)

        token_ids = seq.block(info.logical_block_idx)
        if len(token_ids) == self.block_size:
            current_hash = self.compute_hash(token_ids, prefix_hash)
            block.update(current_hash, token_ids.copy())
            self.hash_to_block_id[current_hash] = gpu_block_id
            prefix_hash = current_hash

        info.residency = BlockResidency.GPU
        info.gpu_block_id = gpu_block_id
        self.free_cpu_slot_ids.append(info.cpu_slot_id)
        info.cpu_slot_id = None
        self.gpu_block_owners[gpu_block_id] = (seq.seq_id, info.logical_block_idx)
        self.blocks[gpu_block_id].residency = BlockResidency.GPU
        new_block_table.append(gpu_block_id)

    seq.block_table = new_block_table
    return new_block_table
```

---

## 5. `engine/model_runner.py`：CPU KV buffer 与每层 block copy

`BlockManager` 决定“哪个 sequence 的哪个 logical block 要换出/换回”，`ModelRunner` 负责真实 tensor copy。当前 `kv_cache` 是每层一个 GPU tensor：

```python
[2, num_blocks, block_size, num_kv_heads, head_dim]
```

Day13 增加一份 CPU backing store，形状一致，只是第二维使用 `num_cpu_blocks`：

```python
self.cpu_kv_cache: Optional[list[torch.Tensor]] = None
```

### 5.1 扩展字段

在 `ModelRunner.__init__()` 的 KV cache 字段附近加入：

```python
self.kv_cache: Optional[list[torch.Tensor]] = None
self.cpu_kv_cache: Optional[list[torch.Tensor]] = None
```

### 5.2 扩展 `allocate_kv_cache`

把 `allocate_kv_cache()` 改成支持 CPU KV buffer，但默认调用 `allocate_kv_cache(num_blocks)` 仍只分配 GPU KV cache。只有显式传入 `enable_cpu_kv_offload=True` 时，才分配 CPU backing store。

```python
def allocate_kv_cache(
    self,
    num_blocks: int,
    num_cpu_blocks: int | None = None,
    enable_cpu_kv_offload: bool = False,
):
    """预分配 GPU KV Cache，并按需分配 Day13 CPU KV backing store。"""
    cpu_blocks = num_cpu_blocks if num_cpu_blocks is not None else num_blocks

    bytes_per_block = (
        2
        * self.block_size
        * self.num_kv_heads
        * self.head_dim
        * 2
    )
    total_bytes = self.num_layers * num_blocks * bytes_per_block
    print(f"[ModelRunner] KV Cache 显存需求：{total_bytes / 1024**3:.2f} GB")

    self.kv_cache = []
    self.cpu_kv_cache = [] if enable_cpu_kv_offload else None

    for _ in range(self.num_layers):
        gpu_cache = torch.zeros(
            2,
            num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        self.kv_cache.append(gpu_cache)

        if enable_cpu_kv_offload:
            assert self.cpu_kv_cache is not None
            cpu_cache = torch.zeros(
                2,
                cpu_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=torch.float16,
                device="cpu",
                pin_memory=torch.cuda.is_available(),
            )
            self.cpu_kv_cache.append(cpu_cache)

    if enable_cpu_kv_offload:
        print(
            f"[ModelRunner] KV Cache 分配完成："
            f"GPU {num_blocks} 块 × {self.num_layers} 层, "
            f"CPU {cpu_blocks} 块 × {self.num_layers} 层"
        )
    else:
        print(f"[ModelRunner] KV Cache 分配完成：GPU {num_blocks} 块 × {self.num_layers} 层")
```

### 5.3 新增 GPU -> CPU copy helper

```python
def copy_kv_block_to_cpu(self, gpu_block_id: int, cpu_slot_id: int) -> None:
    """把每一层的一个 KV block 从 GPU cache copy 到 CPU buffer。"""
    assert self.kv_cache is not None, "KV cache is not allocated"
    assert self.cpu_kv_cache is not None, "CPU KV cache is not allocated"

    for layer_idx, gpu_cache in enumerate(self.kv_cache):
        src = gpu_cache[:, gpu_block_id]
        dst = self.cpu_kv_cache[layer_idx][:, cpu_slot_id]
        dst.copy_(
            src.detach().to(device="cpu", dtype=dst.dtype),
            non_blocking=torch.cuda.is_available(),
        )
```

### 5.4 新增 CPU -> GPU copy helper

```python
def copy_kv_block_to_gpu(self, cpu_slot_id: int, gpu_block_id: int) -> None:
    """把每一层的一个 KV block 从 CPU buffer copy 回 GPU cache。"""
    assert self.kv_cache is not None, "KV cache is not allocated"
    assert self.cpu_kv_cache is not None, "CPU KV cache is not allocated"

    for layer_idx, gpu_cache in enumerate(self.kv_cache):
        dst = gpu_cache[:, gpu_block_id]
        src = self.cpu_kv_cache[layer_idx][:, cpu_slot_id].to(
            device=dst.device,
            dtype=dst.dtype,
            non_blocking=torch.cuda.is_available(),
        )
        dst.copy_(src, non_blocking=torch.cuda.is_available())
```

### 5.5 把 copy helper 注册给 `BlockManager`

在 `engine/model_runner.py` 顶部加入类型导入：

```python
from engine.block_manager import BlockManager
```

在 `ModelRunner` 中新增：

```python
def attach_kv_offload(self, block_manager: BlockManager) -> None:
    """把 ModelRunner 的真实 KV copy 函数注册给 BlockManager。"""
    block_manager.register_kv_swap_handlers(
        copy_gpu_block_to_cpu=self.copy_kv_block_to_cpu,
        copy_cpu_block_to_gpu=self.copy_kv_block_to_gpu,
    )
```

默认主线既不传 `enable_cpu_kv_offload=True`，也不调用 `attach_kv_offload()`，因此不会分配 `cpu_kv_cache`，也不会触发任何 swap copy。Day13 测试可以用 fake copy handler 或 `object.__new__(ModelRunner)` 验证 copy helper，不需要加载真实模型。

---

## 6. `engine/scheduler.py`：新增 swapped 队列与换入换出调度

### 6.1 扩展生命周期说明

把 `Scheduler` docstring 中的生命周期从：

```text
WAITING -> RUNNING -> FINISHED
```

扩展为：

```text
WAITING -> RUNNING -> SWAPPED -> RUNNING -> FINISHED
```

其中 `SWAPPED` 不表示请求失败，也不表示需要重新 prefill；它只表示 GPU 侧 KV blocks 已释放，CPU 侧仍有 KV backing store。

### 6.2 新增 `swapped` 队列

在 `Scheduler.__init__()` 中加入：

```python
self.waiting: deque[Sequence] = deque()
self.running: deque[Sequence] = deque()
self.swapped: deque[Sequence] = deque()
```

把 `is_finished()` 改为：

```python
def is_finished(self) -> bool:
    """检查请求是否都已完成。"""
    return len(self.waiting) == 0 and len(self.running) == 0 and len(self.swapped) == 0
```

### 6.3 新增队列操作 helper

在 `Scheduler` 中加入下面两个方法：

```python
def swap_out_sequence(self, seq: Sequence) -> bool:
    """把 running sequence 换出到 CPU，并移动到 swapped 队列。"""
    if not self.block_manager.can_swap_out(seq):
        return False

    try:
        self.running.remove(seq)
    except ValueError:
        pass

    self.block_manager.swap_out(seq)
    seq.status = SequenceStatus.SWAPPED
    self.swapped.append(seq)
    return True


def try_swap_in(self) -> bool:
    """尝试把 swapped 队列头部 sequence 换回 GPU，并移动到 running 队列。"""
    if not self.swapped:
        return False

    seq = self.swapped[0]
    if not self.block_manager.can_swap_in(seq):
        return False

    self.block_manager.swap_in(seq)
    seq.status = SequenceStatus.RUNNING
    self.swapped.popleft()
    self.running.append(seq)
    return True
```

### 6.4 替换 `schedule()`

把 `schedule()` 替换为下面版本。它保留 prefill 优先策略；区别是：当 GPU KV blocks 不够时，优先尝试把 running 队列尾部的可换出 sequence 放入 `SWAPPED`，而不是直接把它 preempt 回 `WAITING`。

```python
def schedule(self) -> Tuple[List[Sequence], bool]:
    """
    核心调度方法。

    Returns:
        (scheduled_seqs, is_prefill)
        - scheduled_seqs: 本次要处理的序列列表
        - is_prefill: True 表示 Prefill 阶段，False 表示 Decode 阶段
    """
    scheduled_seqs: List[Sequence] = []
    num_seqs = 0
    num_batched_tokens = 0

    # ===== 阶段1：Prefill 优先 =====
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        new_tokens = len(seq) - seq.num_cached_tokens
        if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
            break

        while not self.block_manager.can_allocate(seq):
            victim = self.block_manager.select_swap_out_victim(self.running)
            if victim is None:
                break
            self.swap_out_sequence(victim)

        if not self.block_manager.can_allocate(seq):
            break

        self.block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)

        num_seqs += 1
        num_batched_tokens += new_tokens

    if scheduled_seqs:
        return scheduled_seqs, True

    # ===== 阶段2：Decode 前尝试换回 CPU KV blocks =====
    while self.swapped and len(self.running) < self.max_num_seqs:
        if not self.try_swap_in():
            break

    # ===== 阶段3：Decode running 队列 =====
    decoded_seqs: List[Sequence] = []

    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        while not self.block_manager.can_append(seq):
            victim = self.block_manager.select_swap_out_victim(self.running)
            if victim is not None:
                self.swap_out_sequence(victim)
                continue

            if self.block_manager.can_swap_out(seq):
                self.swap_out_sequence(seq)
                seq = None
                break

            self.__preempt(seq)
            seq = None
            break

        if seq is None:
            continue

        self.block_manager.append_slot(seq)
        decoded_seqs.append(seq)
        num_seqs += 1

    for seq in reversed(decoded_seqs):
        self.running.appendleft(seq)

    return decoded_seqs, False
```

### 6.5 保留 preempt fallback

`__preempt()` 保留原语义：当 offload 未启用、没有 copy handler、sequence 共享 prefix block、CPU slots 不够，或当前 sequence 无法安全换出时，仍然释放 KV 并回到 `WAITING`。

```python
def __preempt(self, seq: Sequence):
    """抢占序列：释放其 KV Cache，下次调度时重新 Prefill。"""
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

### 6.6 更新 `postprocess()` 和统计方法

`postprocess()` 在删除 finished seq 时，仍然释放所有资源；为了避免队列中不存在时报错，删除 running 队列元素时加存在性判断。

```python
if is_eos or is_max_tokens:
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    if seq in self.running:
        self.running.remove(seq)
    finished_seqs.append(seq)
```

新增 swapped 队列长度查询：

```python
def get_num_swapped(self) -> int:
    """获取已换出队列长度。"""
    return len(self.swapped)
```

更新 `__repr__()`：

```python
def __repr__(self) -> str:
    return (
        f"Scheduler(waiting={self.get_num_waiting()}, "
        f"running={self.get_num_running()}, "
        f"swapped={self.get_num_swapped()}, "
        f"free_blocks={self.block_manager.get_num_free_blocks()})"
    )
```

---

## 7. `tests/test_Day13_kv_offload.py`：完整测试文件

创建 `tests/test_Day13_kv_offload.py`，内容如下。这个测试文件不加载真实大模型，不读取 Hugging Face 权重；它只使用 `Sequence`、`BlockManager`、`Scheduler` 和通过 `object.__new__(ModelRunner)` 构造出的轻量 copy helper 测试对象。

```python
"""Day13 CPU KV block offload 教学测试。

运行方式：
python tests/test_Day13_kv_offload.py
"""

from contextlib import contextmanager
from dataclasses import dataclass
import sys

import torch

sys.path.insert(0, ".")

from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager, BlockResidency
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner


@contextmanager
def temporary_sequence_block_size(block_size: int):
    old_block_size = Sequence.block_size
    Sequence.block_size = block_size
    try:
        yield
    finally:
        Sequence.block_size = old_block_size


@dataclass
class DummyConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 4096
    eos: int = -1


class CopyRecorder:
    def __init__(self):
        self.gpu_to_cpu: list[tuple[int, int]] = []
        self.cpu_to_gpu: list[tuple[int, int]] = []

    def copy_gpu_block_to_cpu(self, gpu_block_id: int, cpu_slot_id: int) -> None:
        self.gpu_to_cpu.append((gpu_block_id, cpu_slot_id))

    def copy_cpu_block_to_gpu(self, cpu_slot_id: int, gpu_block_id: int) -> None:
        self.cpu_to_gpu.append((cpu_slot_id, gpu_block_id))


def build_block_manager(num_blocks: int = 4, block_size: int = 2):
    recorder = CopyRecorder()
    block_manager = BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        enable_cpu_kv_offload=True,
        num_cpu_blocks=num_blocks,
    )
    block_manager.register_kv_swap_handlers(
        copy_gpu_block_to_cpu=recorder.copy_gpu_block_to_cpu,
        copy_cpu_block_to_gpu=recorder.copy_cpu_block_to_gpu,
    )
    return block_manager, recorder


def test_sequence_status_has_swapped_state():
    seq = Sequence([1, 2, 3])
    seq.status = SequenceStatus.SWAPPED

    assert seq.status == SequenceStatus.SWAPPED
    assert seq.is_swapped
    assert not seq.is_finished


def test_block_residency_metadata_tracks_gpu_then_cpu():
    with temporary_sequence_block_size(2):
        block_manager, _ = build_block_manager(num_blocks=4, block_size=2)
        seq = Sequence([10, 11, 12])

        block_manager.allocate(seq)

        assert len(seq.block_table) == 2
        assert block_manager.get_residency(seq, 0).residency == BlockResidency.GPU
        assert block_manager.get_residency(seq, 1).residency == BlockResidency.GPU

        block_manager.swap_out(seq)

        assert seq.block_table == []
        assert block_manager.get_residency(seq, 0).residency == BlockResidency.CPU
        assert block_manager.get_residency(seq, 1).residency == BlockResidency.CPU
        assert block_manager.get_num_free_blocks() == 4


def test_swap_out_and_swap_in_call_copy_handlers_and_rebuild_block_table():
    with temporary_sequence_block_size(2):
        block_manager, recorder = build_block_manager(num_blocks=4, block_size=2)
        seq = Sequence([20, 21, 22])

        block_manager.allocate(seq)
        old_block_table = seq.block_table.copy()

        returned_old_table = block_manager.swap_out(seq)
        assert returned_old_table == old_block_table
        assert seq.block_table == []
        assert len(recorder.gpu_to_cpu) == 2
        assert recorder.cpu_to_gpu == []

        new_block_table = block_manager.swap_in(seq)
        assert seq.block_table == new_block_table
        assert len(seq.block_table) == 2
        assert len(recorder.cpu_to_gpu) == 2
        assert block_manager.get_residency(seq, 0).residency == BlockResidency.GPU
        assert block_manager.get_residency(seq, 1).residency == BlockResidency.GPU


def test_model_runner_kv_copy_helpers_move_one_block_without_loading_model():
    runner = object.__new__(ModelRunner)
    runner.device = torch.device("cpu")
    runner.kv_cache = [torch.zeros(2, 3, 2, 1, 1, dtype=torch.float16)]
    runner.cpu_kv_cache = [torch.zeros(2, 3, 2, 1, 1, dtype=torch.float16, device="cpu")]

    runner.kv_cache[0][:, 0].fill_(7)
    runner.copy_kv_block_to_cpu(gpu_block_id=0, cpu_slot_id=2)
    assert torch.allclose(runner.cpu_kv_cache[0][:, 2], torch.full((2, 2, 1, 1), 7, dtype=torch.float16))

    runner.cpu_kv_cache[0][:, 2].fill_(9)
    runner.copy_kv_block_to_gpu(cpu_slot_id=2, gpu_block_id=1)
    assert torch.allclose(runner.kv_cache[0][:, 1], torch.full((2, 2, 1, 1), 9, dtype=torch.float16))


def test_scheduler_moves_running_sequence_to_swapped_queue_and_back():
    with temporary_sequence_block_size(2):
        block_manager, recorder = build_block_manager(num_blocks=4, block_size=2)
        scheduler = Scheduler(DummyConfig(), block_manager)
        seq = Sequence([30, 31, 32])

        block_manager.allocate(seq)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        did_swap_out = scheduler.swap_out_sequence(seq)
        assert did_swap_out
        assert seq.status == SequenceStatus.SWAPPED
        assert scheduler.get_num_running() == 0
        assert scheduler.get_num_swapped() == 1
        assert len(recorder.gpu_to_cpu) == 2

        did_swap_in = scheduler.try_swap_in()
        assert did_swap_in
        assert seq.status == SequenceStatus.RUNNING
        assert scheduler.get_num_running() == 1
        assert scheduler.get_num_swapped() == 0
        assert len(recorder.cpu_to_gpu) == 2


def run_all_tests():
    test_sequence_status_has_swapped_state()
    test_block_residency_metadata_tracks_gpu_then_cpu()
    test_swap_out_and_swap_in_call_copy_handlers_and_rebuild_block_table()
    test_model_runner_kv_copy_helpers_move_one_block_without_loading_model()
    test_scheduler_moves_running_sequence_to_swapped_queue_and_back()


if __name__ == "__main__":
    run_all_tests()
    print("Day13 CPU KV block offload tests passed")
```

---

## 8. 验收命令

应用代码块后，从 `nano_vll_repro/` 运行：

```bash
python -m py_compile engine/sequence.py engine/block_manager.py engine/model_runner.py engine/scheduler.py tests/test_Day13_kv_offload.py
python tests/test_Day13_kv_offload.py
```

第一条验证语法，第二条验证 `SWAPPED` 状态、residency 元数据、swap-out/swap-in copy handler、CPU/GPU KV block copy、swapped 队列状态——全部不需要加载真实大模型。

---

## 9. 常见坑

1. **把 CPU KV block offload 写成默认路径。** `enable_cpu_kv_offload` 默认必须是 `False`，主线行为保持稳定。

2. **从 `layers/attention.py` 开始改。** Attention 只消费 `Context.block_tables` 和 `kv_cache`。重点是先让调度层保证：进入 decode 前，所需 block 已经在 GPU cache 中。

3. **把 shared prefix block 作为换出候选。** 教学补丁只选 `ref_count == 1` 的 block。共享 block 的多 owner residency 账本会显著增加复杂度。

4. **把 CPU slot 和 GPU block id 混成同一个概念。** 换出后旧 GPU block id 释放给其他 sequence；CPU 侧数据由 `cpu_slot_id` 管理。换回时重新分配 GPU block id，重建 `seq.block_table`。

5. **引入生产级 offload 组件。** production KV connector、LMCache、async prefetch、JAX、MLX、C++ 后端都不做。学习重点是 block residency、KV copy、队列状态和最小可测语义。

---

## 10. 读完你应该明白

1. `SWAPPED` 与 `WAITING` 的区别：`SWAPPED` 保留 CPU KV backing store，不需要重新 prefill；`WAITING` 表示还没有可复用 KV，调度时要走 prefill。
2. `BlockManager` 除了管理“哪个 logical block 映射到哪个 GPU block”，还可以管理“这个 logical block 当前驻留在 CPU 还是 GPU”。
3. `ModelRunner` 是 KV tensor copy 的正确边界，因为它持有每层 `kv_cache` tensor。
4. `Scheduler` 是状态转换的正确边界，因为它决定 sequence 何时 running、何时 swapped、何时换回。
5. 对这个学习项目来说，CPU KV block offload 的最小闭环是：状态枚举、residency 元数据、copy helper、swapped 队列和不依赖真实模型的测试。
