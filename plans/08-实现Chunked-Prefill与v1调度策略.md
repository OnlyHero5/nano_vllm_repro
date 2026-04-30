# 08. 实现 Chunked Prefill 与 v1 调度策略

这一篇开始吸收主线 `00~07` 之外的新能力，但仍然必须站在当前仓库的真实代码边界上。

这一篇只做两件事：

> 1. 把当前“整段 prompt 一次性 prefill”的调度方式，升级成“可以分块推进的 chunked prefill”。
> 2. 让 `Scheduler` 和 `ModelRunner` 能围绕“还有多少 prefill token 没算完”来协同工作。

这一篇不做下面这些事：

- 不引入 speculative decoding。
- 不引入 MoE。
- 不引入新的 Attention kernel。
- 不改 `layers/attention.py` 的 FlashAttention 调用协议。
- 不把当前 `Sequence` 变成另一个完全不同的请求对象。

原因很简单：

> `chunked prefill` 的核心不是换模型，也不是换 kernel，而是把“prefill 这件事”从“一次性把整段 prompt 全算完”改成“按照 token budget 分段推进”。

---

## 1. 为什么现在要补 chunked prefill

当前仓库在 `engine/scheduler.py` 里的 prefill 路径是“整条序列优先、整段 prompt 一次性上车”。核心判断是：

```python
new_tokens = len(seq) - seq.num_cached_tokens
if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
    break
```

它的优点是简单，缺点也很明显：

1. **长 prompt 容易卡住 batch。**
   如果 waiting 队列头部有一条很长的请求，它可能直接吃掉本轮大部分 token budget。
2. **prefill 和 decode 的公平性较差。**
   新请求一旦足够长，会持续拖住 decode 请求，导致 running 队列里已经在生成的序列得不到足够执行机会。
3. **prefix cache 命中时虽然会减少 `new_tokens`，但仍然是“按整条剩余 prompt 一次性尝试调度”。**

`chunked prefill` 的思路是：

> 不再要求“一个 prefill 请求必须一次算完剩余所有 prompt token”，而是允许它在本轮只处理其中一个 chunk，下一轮再继续处理剩下的 prompt token。

这和社区里 2026 年前后的 nano-vllm 扩展仓、以及 vLLM v1 风格调度思路是一致的：

- token budget 是本轮的核心资源。
- prefill 请求可以被切成多个 prefill step。
- decode 不再总是被长 prompt 挤压。

---

## 2. 当前代码是什么状态

当前与 prefill 直接相关的地方有四个：

1. `engine/sequence.py`
2. `engine/scheduler.py`
3. `engine/model_runner.py`
4. `engine/llm_engine.py`

### 2.1 `Sequence` 当前只有“总 token”和“已缓存 token”

当前 `Sequence` 已经有：

- `token_ids`
- `num_prompt_tokens`
- `num_cached_tokens`
- `block_table`

这很好，因为它已经能表达：

> 这条请求一共有多少 token，其中有多少 token 已经进入 KV Cache。

但当前它还没有一个明确的“本轮 prefill 该处理到哪”的辅助接口。

### 2.2 `Scheduler.schedule()` 现在是整条 prompt 一次性尝试装入

现在的 waiting 路径是：

- 取队首序列。
- 算 `new_tokens = len(seq) - seq.num_cached_tokens`。
- 如果剩余 prompt token 总数塞不下，就直接 `break`。

这就意味着：

> waiting 队列头部那条请求，如果剩余 prompt 太长，会直接卡住后面更短的请求。

### 2.3 `ModelRunner.prepare_prefill()` 现在总是把整条 `seq.token_ids` 全部送进模型

当前版本会这样做：

```python
token_ids = seq.token_ids
seq_len = len(token_ids)
all_token_ids.extend(token_ids)
all_positions.extend(range(seq_len))
```

这对应的是“全量 prefill”，而不是“只 prefill 还没算过的那一段 chunk”。

### 2.4 `LLMEngine.step()` 的吞吐统计也还是按“整条 prefill”口径在想

所以这篇必须同时改调度与输入准备，否则会出现一个典型错误：

> 调度器以为自己切 chunk 了，`ModelRunner` 却还是把整条 prompt 重新送进模型。

---

## 3. 先明确 chunked prefill 的新账本

在当前仓库里，我们不新增复杂请求对象，只在 `Sequence` 上补齐一组清楚的运行时语义。

### 3.1 必须分清三个数字

对每条序列，后面要长期区分这三个量：

1. `num_prompt_tokens`：原始 prompt 长度。
2. `num_cached_tokens`：已经进入 KV Cache 的 prompt token 数。
3. `num_uncomputed_tokens`：还没做 prefill 计算的 prompt token 数。

在当前仓库语义里：

```python
num_uncomputed_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
```

注意，这里只看 prompt 部分。decode 生成出来的新 token 不属于“等待 prefill 的 prompt token”。

### 3.2 chunked prefill 的关键不是拆 block，而是拆本轮输入窗口

很多人第一次看 chunked prefill，会误以为：

> 是不是要把 `Sequence.token_ids` 真正切成几个子序列对象？

这里不需要。更稳的做法是：

- `Sequence.token_ids` 继续保存完整 token 序列。
- `num_cached_tokens` 继续表示“prompt 前缀里已经进 cache 的部分”。
- 每一轮由调度器决定这条请求这次最多再前进多少个 prompt token。

也就是说：

> chunked prefill 更像“给现有序列指定一个本轮处理窗口”，而不是“重新发明序列对象”。

---

## 4. 修改 `engine/sequence.py`

### 4.1 增加 chunked prefill 辅助属性

下面是可以直接加入 `Sequence` 类的完整方法块。它不改变当前类的构造协议，只补充 prompt 账本查询能力。

```python
@property
def num_prompt_blocks(self) -> int:
    """
    prompt 本身一共会占用多少逻辑块。

    chunked prefill 需要区分“prompt 长度”和“当前总长度”，
    因为 decode 生成出来的新 token 不属于等待 prefill 的 prompt。
    """
    return (self.num_prompt_tokens + self.block_size - 1) // self.block_size


@property
def num_uncomputed_tokens(self) -> int:
    """
    还有多少 prompt token 没做 prefill 计算。

    这里必须只看 prompt 段，不能把 decode 生成出来的新 token 算进去。
    """
    return max(0, self.num_prompt_tokens - self.num_cached_tokens)


@property
def prefill_done(self) -> bool:
    """
    当前 prompt 是否已经全部完成 prefill。
    """
    return self.num_cached_tokens >= self.num_prompt_tokens


def get_chunk_token_ids(self, chunk_size: int) -> list[int]:
    """
    返回当前轮需要执行 prefill 的那一段 prompt token。

    约定：
    - chunk 一定从 num_cached_tokens 开始。
    - chunk 只覆盖 prompt 段，不覆盖 decode 生成段。
    - 如果剩余 prompt 不足 chunk_size，就返回剩余全部。
    """
    assert chunk_size > 0, "chunk_size 必须 > 0"

    start = self.num_cached_tokens
    end = min(self.num_prompt_tokens, start + chunk_size)
    return self.prompt_token_ids[start:end]
```

### 4.2 这一组方法解决了什么问题

它真正解决的是：

1. `Scheduler` 能直接问：这条序列还剩多少 prompt token 没算？
2. `ModelRunner` 能直接拿到：本轮只该送进模型的那一段 chunk 是什么？
3. 文档和代码都不再把 `num_cached_tokens` 误解释成“整个序列已经全算过多少 token”。

---

## 5. 修改 `Config`

为了让 `chunked prefill` 真正可控，建议在 `config.py` 的 `Config` 数据类里增加一个参数：

```python
max_prefill_chunk_size: int = 1024
```

它的含义是：

> 单条序列在一次 prefill step 里最多处理多少个未计算 prompt token。

这个参数不应该替代 `max_num_batched_tokens`。两者区别是：

- `max_prefill_chunk_size` 是单条序列每次最多推进多少。
- `max_num_batched_tokens` 是整个 batch 本轮最多处理多少。

---

## 6. 修改 `engine/scheduler.py`

### 6.1 调度器必须返回“本轮 prefill 处理多少 token”

如果 `schedule()` 仍然只返回 `list[Sequence]`，`ModelRunner` 就不知道每条 waiting 序列这轮究竟该处理多少 prompt token。

所以推荐把调度结果改成三元组：

```python
scheduled_seqs, is_prefill, prefill_chunk_sizes = scheduler.schedule()
```

其中：

- `scheduled_seqs`：本轮要处理的序列。
- `is_prefill`：是否走 prefill 路径。
- `prefill_chunk_sizes`：仅在 prefill 时使用，表示每条序列这轮该处理多少个 prompt token。

### 6.2 推荐完整实现

下面是可以直接替换当前 `Scheduler.schedule()` 并新增 `mark_prefill_progress()` 的完整代码块。

```python
from typing import List, Tuple


def schedule(self) -> Tuple[List[Sequence], bool, List[int]]:
    """
    返回：
    (scheduled_seqs, is_prefill, prefill_chunk_sizes)

    约定：
    - prefill 路径下，prefill_chunk_sizes 与 scheduled_seqs 一一对应。
    - decode 路径下，prefill_chunk_sizes 返回空列表。
    """
    scheduled_seqs: List[Sequence] = []
    prefill_chunk_sizes: List[int] = []
    num_seqs = 0
    num_batched_tokens = 0
    max_prefill_chunk_size = getattr(self, "max_prefill_chunk_size", 1024)

    # ===== 阶段 1：chunked prefill =====
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        if not seq.block_table and not self.block_manager.can_allocate(seq):
            break

        remaining_prompt_tokens = seq.num_uncomputed_tokens
        if remaining_prompt_tokens <= 0:
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            continue

        remaining_batch_budget = self.max_num_batched_tokens - num_batched_tokens
        chunk_size = min(remaining_prompt_tokens, max_prefill_chunk_size, remaining_batch_budget)
        if chunk_size <= 0:
            break

        if not seq.block_table:
            self.block_manager.allocate(seq)

        self.waiting.popleft()
        scheduled_seqs.append(seq)
        prefill_chunk_sizes.append(chunk_size)
        num_seqs += 1
        num_batched_tokens += chunk_size

        # 如果这条序列本轮不会完成 prefill，放到队尾，让后续短请求也有机会被调度。
        if chunk_size < remaining_prompt_tokens:
            self.waiting.append(seq)
        else:
            # 本轮完成 prompt prefill 后，账本更新函数会把它转入 running。
            # 这里先保留在本轮 scheduled 列表里，不再放回 waiting。
            pass

    if scheduled_seqs:
        return scheduled_seqs, True, prefill_chunk_sizes

    # ===== 阶段 2：decode =====
    decoded_seqs: List[Sequence] = []
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        while not self.block_manager.can_append(seq):
            if self.running:
                victim = self.running.pop()
                self.__preempt(victim)
            else:
                self.__preempt(seq)
                break
        else:
            self.block_manager.append_slot(seq)
            decoded_seqs.append(seq)
            num_seqs += 1

    for seq in reversed(decoded_seqs):
        self.running.appendleft(seq)

    return decoded_seqs, False, []


def mark_prefill_progress(self, seqs: List[Sequence], chunk_sizes: List[int]) -> None:
    """
    在一轮 prefill 完成后，把本轮已经算完并写入 cache 的 prompt token 记到账本里。
    """
    assert len(seqs) == len(chunk_sizes), "seqs 与 chunk_sizes 必须一一对应"

    for seq, chunk_size in zip(seqs, chunk_sizes):
        seq.num_cached_tokens = min(seq.num_prompt_tokens, seq.num_cached_tokens + chunk_size)

        if seq.prefill_done:
            seq.status = SequenceStatus.RUNNING
            if seq not in self.running:
                self.running.append(seq)
        else:
            seq.status = SequenceStatus.WAITING
            if seq not in self.waiting:
                self.waiting.append(seq)
```

### 6.3 在 `Scheduler.__init__()` 里接入参数

```python
self.max_prefill_chunk_size = getattr(config, "max_prefill_chunk_size", 1024)
```

这一步很小，但很关键。没有它，chunk 大小会变成写死的隐藏常量。

---

## 7. 修改 `engine/model_runner.py`

### 7.1 `prepare_prefill()` 必须只处理本轮 chunk

当前版本会直接把整条 `seq.token_ids` 全部喂进去，这和 chunked prefill 冲突。

下面是可以直接替换 `prepare_prefill()` 的完整方法块。

```python
def prepare_prefill(
    self,
    sequences: list[Sequence],
    chunk_sizes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    准备 chunked prefill 阶段的输入。

    关键变化：
    - 每条序列这轮只处理一个 chunk。
    - positions 不是从 0 开始，而是从 num_cached_tokens 开始。
    - slot_mapping 只覆盖这轮新写入 cache 的那一段 token。
    """
    assert len(sequences) == len(chunk_sizes), "sequences 与 chunk_sizes 必须一一对应"

    all_token_ids: list[int] = []
    all_positions: list[int] = []
    cu_seqlens = [0]
    slot_mapping: list[int] = []

    for seq, chunk_size in zip(sequences, chunk_sizes):
        chunk_token_ids = seq.get_chunk_token_ids(chunk_size)
        start_pos = seq.num_cached_tokens
        end_pos = start_pos + len(chunk_token_ids)

        all_token_ids.extend(chunk_token_ids)
        all_positions.extend(range(start_pos, end_pos))
        cu_seqlens.append(cu_seqlens[-1] + len(chunk_token_ids))

        for pos in range(start_pos, end_pos):
            block_idx = pos // self.block_size
            offset = pos % self.block_size

            if block_idx < len(seq.block_table):
                block_id = seq.block_table[block_idx]
                slot_mapping.append(block_id * self.block_size + offset)
            else:
                slot_mapping.append(0)

    input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=self.device)
    positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)
    max_seqlen = max(chunk_sizes) if chunk_sizes else 0

    context = Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
        cu_seqlens_k=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
        context_lens=None,
        block_tables=None,
        max_context_len=None,
        max_num_blocks=None,
        kv_cache=self.kv_cache,
    )
    set_context(context)
    return input_ids, positions
```

### 7.2 `run()` 必须接收 `chunk_sizes`

下面是可以直接替换 `ModelRunner.run()` 的完整方法块。它假设前面 `04` 已经把 `run_model()` 拆出来，并且 `03` 已经让 sampler 支持 `top_k / top_p`。

```python
@torch.inference_mode()
def run(
    self,
    sequences: list[Sequence],
    is_prefill: bool,
    chunk_sizes: list[int] | None = None,
) -> list[int]:
    """
    执行一个 step。

    关键点：
    - prefill 时必须显式传 chunk_sizes。
    - decode 时 chunk_sizes 保持 None。
    - 不管本轮是否报错，Context 都要在 finally 里清掉。
    """
    if not sequences:
        return []

    if is_prefill:
        assert chunk_sizes is not None, "prefill 路径必须提供 chunk_sizes"
        input_ids, positions = self.prepare_prefill(sequences, chunk_sizes)
    else:
        input_ids, positions = self.prepare_decode(sequences)

    try:
        logits = self.run_model(input_ids, positions, is_prefill)
        temperatures = torch.tensor(
            [seq.temperature for seq in sequences],
            dtype=torch.float32,
            device=self.device,
        )
        top_ks = torch.tensor(
            [getattr(seq, "top_k", 0) for seq in sequences],
            dtype=torch.long,
            device=self.device,
        )
        top_ps = torch.tensor(
            [getattr(seq, "top_p", 1.0) for seq in sequences],
            dtype=torch.float32,
            device=self.device,
        )
        next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
        return next_tokens.tolist()
    finally:
        reset_context()
```

### 7.3 这一版最容易看漏的点

最容易看漏的是 `positions`。

在 chunked prefill 里，第二个 chunk 的位置绝不能重新从 0 开始。

比如一条 prompt 已经 prefill 了前 512 个 token，本轮再处理下一个 256-token chunk，那么本轮 positions 应该是：

```python
list(range(512, 768))
```

而不是：

```python
list(range(256))
```

如果这里写错，RoPE 和 KV 写入位置都会一起错。

---

## 8. 修改 `engine/llm_engine.py`

`LLMEngine.step()` 必须显式传递 chunk 信息。下面是可以直接替换 `step()` 的完整方法块。

```python
def step(self) -> tuple[list[tuple[int, list[int]]], int]:
    """
    执行一次调度和推理。

    num_tokens 约定：
    - prefill 阶段返回正数，表示本轮新计算的 prompt token 数。
    - decode 阶段返回负数，绝对值表示本轮处理了多少条序列。
    """
    seqs, is_prefill, chunk_sizes = self.scheduler.schedule()
    if not seqs:
        return [], 0

    if is_prefill:
        num_tokens = sum(chunk_sizes)
        token_ids = self.model_runner.run(seqs, True, chunk_sizes)
        self.scheduler.mark_prefill_progress(seqs, chunk_sizes)
        finished_seqs = self.scheduler.postprocess(seqs, token_ids)
    else:
        num_tokens = -len(seqs)
        token_ids = self.model_runner.run(seqs, False)
        finished_seqs = self.scheduler.postprocess(seqs, token_ids)

    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
    return outputs, num_tokens
```

### 8.1 为什么 prefill 后还要走 `postprocess()`

因为当前教学仓库的主线语义是：

- 一轮模型执行结束。
- sampler 选出下一个 token。
- `postprocess()` 把这个 token append 到序列上。

即使 prefill 这轮只处理了 prompt chunk，它最后仍然会得到每条序列当前 step 的最后 logits，并采样出一个“下一 token”。

所以这一轮结束时，仍然需要：

1. 先记账：本轮 chunk 对应的 prompt token 已经进 cache。
2. 再 append：本轮真正新生成的 decode token。

这两件事不能颠倒。

---

## 9. 新增 `tests/test_Day8_chunked_prefill.py`

这一篇的测试目标，不是跑大模型，而是锁住 chunk 账本和调度边界。

下面这份测试脚本可以直接使用：

```python
"""Day8 chunked prefill 结构测试。"""

import sys

sys.path.insert(0, ".")

import torch

from sampling_params import SamplingParams
from engine.sequence import Sequence, SequenceStatus


def test_sequence_chunk_properties():
    seq = Sequence([10, 11, 12, 13, 14, 15], SamplingParams())

    assert seq.num_prompt_tokens == 6
    assert seq.num_uncomputed_tokens == 6
    assert seq.prefill_done is False

    seq.num_cached_tokens = 2
    assert seq.num_uncomputed_tokens == 4
    assert seq.get_chunk_token_ids(2) == [12, 13]
    assert seq.get_chunk_token_ids(16) == [12, 13, 14, 15]


def test_scheduler_returns_chunk_sizes():
    from config import Config
    from engine.block_manager import BlockManager
    from engine.scheduler import Scheduler

    config = Config(model_path="models/Qwen3-0.6B")
    config.max_num_batched_tokens = 4
    config.max_num_seqs = 2
    config.max_prefill_chunk_size = 2

    block_manager = BlockManager(num_blocks=16, block_size=256)
    scheduler = Scheduler(config, block_manager)

    seq1 = Sequence([1, 2, 3, 4, 5], SamplingParams())
    seq2 = Sequence([6, 7, 8], SamplingParams())

    scheduler.add(seq1)
    scheduler.add(seq2)

    seqs, is_prefill, chunk_sizes = scheduler.schedule()

    assert is_prefill is True
    assert seqs == [seq1, seq2]
    assert chunk_sizes == [2, 2]


def test_mark_prefill_progress_moves_finished_prompt_to_running():
    from config import Config
    from engine.block_manager import BlockManager
    from engine.scheduler import Scheduler

    config = Config(model_path="models/Qwen3-0.6B")
    block_manager = BlockManager(num_blocks=16, block_size=256)
    scheduler = Scheduler(config, block_manager)

    seq = Sequence([1, 2, 3], SamplingParams())
    scheduler.waiting.append(seq)

    scheduler.mark_prefill_progress([seq], [3])

    assert seq.num_cached_tokens == 3
    assert seq.prefill_done is True
    assert seq.status == SequenceStatus.RUNNING
    assert seq in scheduler.running


@torch.inference_mode()
def test_chunked_prefill_step_counts_chunk_tokens_only():
    from engine.llm_engine import LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    seq = Sequence([1, 2, 3, 4, 5, 6], SamplingParams())

    class FakeScheduler:
        def schedule(self):
            return [seq], True, [2]

        def mark_prefill_progress(self, seqs, chunk_sizes):
            assert seqs == [seq]
            assert chunk_sizes == [2]
            seq.num_cached_tokens += 2
            seq.status = SequenceStatus.RUNNING

        def postprocess(self, seqs, token_ids):
            assert token_ids == [99]
            seq.append_token(99)
            seq.status = SequenceStatus.FINISHED
            return [seq]

    class FakeRunner:
        def run(self, seqs, is_prefill, chunk_sizes=None):
            assert seqs == [seq]
            assert is_prefill is True
            assert chunk_sizes == [2]
            return [99]

    engine.scheduler = FakeScheduler()
    engine.model_runner = FakeRunner()

    outputs, num_tokens = engine.step()

    assert num_tokens == 2
    assert outputs == [(seq.seq_id, [99])]
```

---

## 10. 验收命令

```bash
python -m py_compile engine/sequence.py engine/scheduler.py engine/model_runner.py engine/llm_engine.py tests/test_Day8_chunked_prefill.py
python tests/test_Day8_chunked_prefill.py
```

如果你已经把前面主线 `00~07` 跑通，再做一轮轻量手测：

```bash
python - <<'PY'
from engine.sequence import Sequence
from sampling_params import SamplingParams

seq = Sequence(list(range(10)), SamplingParams())
seq.num_cached_tokens = 4

print("remaining:", seq.num_uncomputed_tokens)
print("chunk:", seq.get_chunk_token_ids(3))
PY
```

预期输出应表达：

- 还剩 6 个未计算 prompt token。
- 本轮 chunk 取到索引 4、5、6 对应的 token。

---

## 11. 常见坑

1. **调度器切了 chunk，`prepare_prefill()` 却还是把整条 prompt 全送进模型。**
   这是最常见的“文义上支持 chunk，实际执行仍是全量 prefill”的假实现。
2. **第二个 chunk 的 positions 又从 0 开始。**
   这会直接破坏 RoPE 与 cache slot 位置语义。
3. **把 `num_cached_tokens` 理解成“整条序列都缓存了多少”，而不是“prompt 前缀里已经做完 prefill 的部分”。**
   这会让 decode 后续账本越来越乱。
4. **prefill 完成后不把序列从 waiting 转到 running。**
   这样 decode 永远接不上。
5. **为了做 chunked prefill，一上来就重写 BlockManager。**
   这一篇完全没必要。当前教学版只需要先把调度和输入窗口改正确。

---

## 12. 本篇结束后你应该明白

这一篇最重要的不是记住 `chunked prefill` 这个词。

真正要学会的是：

1. `chunked prefill` 本质上是在改“本轮 token budget 怎么分配”。
2. 当前仓库里，最自然的接入点是 `Sequence -> Scheduler -> ModelRunner -> LLMEngine.step()`。
3. `positions` 和 `slot_mapping` 必须按 chunk 正确推进，否则就是假实现。
4. 这一篇先改调度和账本，不急着碰更复杂的 kernel、spec decode 或 MoE。

下一篇进入 radix / prefix-tree cache：

- `09-实现Radix-Prefix-Cache与可观测指标.md`
