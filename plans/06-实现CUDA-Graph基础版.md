# 06. 实现 CUDA Graph 基础版

这一篇做一版最小 CUDA Graph。

范围先写死：

1. 只支持 CUDA。
2. 只图化 decode。
3. 只做 batch size 精确命中。
4. `tensor_parallel_size > 1` 时回到 eager。
5. sampler 不放进 graph。

目标是：

> decode 阶段命中已录制 batch size 时走 graph replay；没命中就继续 eager。

---

## 1. 前置条件

默认你已经完成：

1. `04`：`ModelRunner` 已经有 `run_model()`。
2. `04`：sampler 已经在 `run_model()` 外面。
3. `05`：`tensor_parallel_size` 和 eager fallback 的边界已经有了。

如果 `run_model()` 还没拆出来，先不要做 CUDA Graph。

---

## 2. 当前代码是什么状态

当前 `utils/context.py` 已经有：

- `Context`
- `set_context()`
- `get_context()`
- `reset_context()`

所以这篇不是补 Context。

这篇要补的是：

1. graph 静态 buffer 放在哪里。
2. capture 什么时候做。
3. replay 前怎么把真实 batch 数据写进静态 buffer。
4. eager 和 replay 结束后都要 `reset_context()`。

---

## 3. 先记住 CUDA Graph 的边界

CUDA Graph 适合重复执行形状固定的计算。

decode 阶段刚好有这个特点：

- 每条序列每轮只输入 1 个 token。
- batch size 命中时，输入张量形状固定。

但下面这些东西不适合先放进 graph：

- sampler：有随机性，参数也动态。
- prefill：prompt 长度差异大。
- TP + graph 联动：基础版先不做。

所以本篇只图化模型主干：

```text
decode input_ids/positions/context
        ↓
model forward
        ↓
hidden states
```

`compute_logits()` 和 sampler 都仍然在 graph 外面。这样边界更简单，后面要不要把 `lm_head` 也录进去，可以单独评估。

---

## 4. 新增 `DecodeGraphRunner`

放在 `engine/model_runner.py` 的 import 后、`class ModelRunner` 前。

```python
from dataclasses import dataclass


@dataclass
class DecodeGraphRunner:
    """
    一个 batch size 对应一套静态 CUDA Graph 资源。

    CUDA Graph replay 时不能换张量对象，
    所以 input_ids、positions、slot_mapping 等都要提前分配好。
    每次 replay 前，只把新 batch 的数据 copy 到这些静态张量里。
    """
    batch_size: int
    max_num_blocks: int

    # 静态输入。
    input_ids: torch.Tensor
    positions: torch.Tensor

    # 静态 Context 相关张量。
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor

    # graph 输出引用。
    hidden_states: torch.Tensor

    # 录制好的 CUDA Graph。
    graph: torch.cuda.CUDAGraph
```

---

## 5. 在 `ModelRunner.__init__()` 里增加 graph 状态

```python
# batch_size -> DecodeGraphRunner
self.decode_graphs: dict[int, DecodeGraphRunner] = {}

# 基础版只在单卡 CUDA 下启用 graph。
self.use_cuda_graph = (
    torch.cuda.is_available()
    and self.device.type == "cuda"
    and not self.config.enforce_eager
    and self.config.tensor_parallel_size == 1
)
```

然后在 `allocate_kv_cache()` 末尾加：

```python
if self.use_cuda_graph:
    self.capture_decode_graphs()
```

为什么 capture 要放在 KV Cache 分配之后：

> decode attention 会读写真实 KV Cache。KV Cache 没分配好时录 graph，图里引用的对象不稳定。

---

## 6. 新增 `capture_decode_graphs()`

```python
@torch.inference_mode()
def capture_decode_graphs(self) -> None:
    """
    录制一组 decode batch size。

    基础版只录小档位：
    1, 2, 4, 8, 16

    注意：
    这里追求的是边界清楚，不是覆盖所有动态形状。
    """
    capture_batch_sizes = [1, 2, 4, 8, 16]
    max_num_blocks = max(1, self.config.max_model_len // self.block_size + 1)

    for batch_size in capture_batch_sizes:
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        context_lens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        block_tables = torch.zeros(
            batch_size,
            max_num_blocks,
            dtype=torch.int32,
            device=self.device,
        )

        # capture 前先设置一份静态 decode Context。
        # Attention 层会通过 get_context() 读取这些张量。
        set_context(
            Context(
                is_prefill=False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                max_context_len=1,
                max_num_blocks=max_num_blocks,
                kv_cache=self.kv_cache,
            )
        )

        # warmup 一次，避免第一次执行的初始化开销进入 graph。
        hidden_states = self.model(input_ids, positions)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden_states = self.model(input_ids, positions)

        self.decode_graphs[batch_size] = DecodeGraphRunner(
            batch_size=batch_size,
            max_num_blocks=max_num_blocks,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            hidden_states=hidden_states,
            graph=graph,
        )

        # 每录完一个档位，都清掉 Context。
        # 不然后面的 eager 路径可能读到旧 graph 的静态 Context。
        reset_context()
```

---

## 7. 替换 `run_model()`

`run_model()` 负责选择 eager 还是 graph replay。

```python
@torch.inference_mode()
def run_model(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_prefill: bool,
) -> torch.Tensor:
    """
    graph-aware 的模型执行入口。

    规则：
    - prefill 永远 eager。
    - 没启用 graph 时 eager。
    - decode batch size 没命中时 eager。
    - 命中时 replay。
    """
    if is_prefill or not self.use_cuda_graph:
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)

        if is_prefill:
            context = get_context()
            last_token_indices = context.cu_seqlens_q[1:] - 1
            logits = logits[last_token_indices.long()]
        return logits

    batch_size = input_ids.shape[0]
    runner = self.decode_graphs.get(batch_size)

    if runner is None:
        # 基础版 exact-match：没录过这个 batch size 就回 eager。
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)

    context = get_context()

    # replay 前，把本轮真实数据写入静态 buffer。
    runner.input_ids.copy_(input_ids)
    runner.positions.copy_(positions)

    runner.slot_mapping.zero_()
    runner.slot_mapping[:batch_size].copy_(context.slot_mapping)

    runner.context_lens.zero_()
    runner.context_lens[:batch_size].copy_(context.context_lens)

    runner.block_tables.zero_()
    runner.block_tables[:batch_size, : context.block_tables.shape[1]].copy_(
        context.block_tables
    )

    # graph replay 时，Attention 仍然通过 Context 读取元数据。
    # 所以要用静态 buffer 重建一份当前 decode Context。
    set_context(
        Context(
            is_prefill=False,
            slot_mapping=runner.slot_mapping,
            context_lens=runner.context_lens,
            block_tables=runner.block_tables,
            max_context_len=int(runner.context_lens.max().item()),
            max_num_blocks=runner.max_num_blocks,
            kv_cache=self.kv_cache,
        )
    )

    runner.graph.replay()
    return self.model.compute_logits(runner.hidden_states)
```

`run()` 不要再塞 graph 分支。它仍然只负责：

1. 准备输入。
2. 准备采样张量。
3. 调 `run_model()`。
4. 调 sampler。
5. `finally: reset_context()`。

---

## 8. 新增 `tests/test_Day6_cudagraph.py`

测试先锁边界，不跑真实 graph。

```python
"""Day6 CUDA Graph 边界测试。"""

import sys
sys.path.insert(0, ".")

import torch

from utils.context import Context, get_context, reset_context, set_context


def test_context_reset_is_hard_boundary():
    # 先写入一个非默认值。
    set_context(Context(is_prefill=False, max_num_blocks=8))
    assert get_context().max_num_blocks == 8

    # reset 后必须回到空 Context。
    reset_context()
    assert get_context().max_num_blocks is None


def test_decode_graph_runner_dataclass_exists():
    from engine.model_runner import DecodeGraphRunner

    assert DecodeGraphRunner is not None


@torch.inference_mode()
def test_graph_is_disabled_on_cpu_or_enforce_eager():
    from config import Config
    from engine.model_runner import ModelRunner

    config = Config(model_path="models/Qwen3-0.6B", enforce_eager=True)

    # 不调用 __init__，避免加载真实模型。
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.device = torch.device("cpu")
    runner.decode_graphs = {}
    runner.kv_cache = None

    runner.use_cuda_graph = (
        torch.cuda.is_available()
        and runner.device.type == "cuda"
        and not runner.config.enforce_eager
        and runner.config.tensor_parallel_size == 1
    )

    assert runner.use_cuda_graph is False
```

---

## 9. 验收命令

```bash
python -m py_compile engine/model_runner.py utils/context.py tests/test_Day6_cudagraph.py
python tests/test_Day6_cudagraph.py
```

如果有 CUDA 环境，并且前面单卡链路已经跑通，再跑：

```bash
python example.py
```

---

## 10. 常见坑

1. **KV Cache 分配前就 capture**
   graph 里会引用不稳定对象。

2. **把 sampler 放进 graph**
   sampler 有随机性和动态参数，基础版先放外面。

3. **replay 前不更新静态 buffer**
   graph 会吃到上一轮 batch 的数据。

4. **eager 和 replay 用两套 Context 协议**
   后面排查 attention 问题会很痛苦。

5. **以为 graph 必须覆盖所有 batch size**
   基础版只做 exact-match，没命中就 eager。

---

## 11. 本篇结束后你应该明白

CUDA Graph 这一篇的重点不是 API 名字，而是四条边界：

1. graph capture 要在 KV Cache 稳定之后。
2. 基础版先做 decode-only。
3. replay 依赖静态 buffer。
4. eager fallback 是正常路径，不是失败。

下一篇进入 benchmark：

- `07-补齐Benchmark与Day7验收.md`
