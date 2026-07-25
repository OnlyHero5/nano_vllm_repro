# Day 8 — Chunked Prefill：别让长 prompt 堵死整个 batch

想象这个场景：waiting 队列头部排着一条 8000 token 的长 prompt，后面跟着三条十几个 token 的短请求。当前调度器的做法是——8000 token 塞不下？那谁也别上车。三条短请求继续干等。

这不太公平。

chunked prefill 的思路很直白：**长 prompt 不必一次全算完，拆成几段，每轮只推进一段。** 这轮算 1024 个 token，下轮再算 1024 个，省下来的 budget 留给 decode 请求。

这次改的是调度和输入准备这条线——模型、kernel、attention 协议都不碰。

---

## 1. 当前 prefill 路径的问题

`engine/scheduler.py` 里的 prefill 判断长这样：

```python
new_tokens = len(seq) - seq.num_cached_tokens
if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
    break
```

简单，但有三个毛病：

1. **长 prompt 卡死 batch。** waiting 队头一条 8000 token 的请求，直接吃掉本轮全部 budget，后面的短请求干瞪眼。
2. **prefill 和 decode 抢资源。** 新请求一长，running 队列里正在生成的序列就被饿着。
3. **prefix cache 命中也没用。** 虽然 `num_cached_tokens` 会减小 `new_tokens`，但调度器仍然是”整条剩余 prompt 一次性尝试塞入”。

chunked prefill 的改法：**允许一条 prefill 请求这轮只处理一部分 prompt token，下轮接着来。** 这和 vLLM v1 的调度思路一致——token budget 是本轮的核心资源，prefill 可以被切成多个 step，decode 不再总被长 prompt 挤压。

---

## 2. 当前代码的四个相关点

和 prefill 直接相关的有四个文件：

### 2.1 `Sequence`：有总账，没细账

当前 `Sequence` 已经有 `token_ids`、`num_prompt_tokens`、`num_cached_tokens`、`block_table`——能表达”一共多少 token，多少已进 cache”。但缺一个关键问题：**本轮该处理到哪？**

### 2.2 `Scheduler.schedule()`：整条 prompt 一次性尝试装入

waiting 路径是：取队首 → 算 `new_tokens` → 塞不下就 `break`。队头那条长请求一旦塞不下，后面更短的请求全部被卡住。

### 2.3 `ModelRunner.prepare_prefill()`：总是送整条 `seq.token_ids`

```python
token_ids = seq.token_ids
seq_len = len(token_ids)
all_token_ids.extend(token_ids)
all_positions.extend(range(seq_len))
```

这是”全量 prefill”，不是”只 prefill 还没算过的那一段”。

### 2.4 `LLMEngine.step()`：吞吐统计也按整条 prefill 在算

所以这篇必须同时改调度和输入准备。只改一头就会出现经典翻车：调度器以为自己切了 chunk，`ModelRunner` 还是把整条 prompt 重新送进模型。

---

## 3. chunked prefill 的账本：三个数字

不新增复杂请求对象，只在 `Sequence` 上补齐一组运行时语义。关键是分清三个量：

1. `num_prompt_tokens`：原始 prompt 长度。
2. `num_cached_tokens`：已进 KV Cache的 prompt token 数。
3. `num_uncomputed_tokens`：还没做 prefill 的 prompt token 数。

```python
num_uncomputed_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
```

只看 prompt 部分——decode 生成的新 token 不算在内。

###3.1 不是拆序列，是拆本轮输入窗口

第一次看 chunked prefill 容易想歪：是不是要把 `token_ids` 切成几个子序列对象？

不用。`token_ids` 继续保存完整序列，`num_cached_tokens` 继续标记”前缀里已进 cache 的部分”，每轮由调度器决定这次最多再前进多少 token。chunked prefill 是给现有序列指定一个本轮处理窗口，不是重新发明序列对象。

---

## 4. 修改 `engine/sequence.py`

### 4.1 增加 chunked prefill 辅助属性

把下面这组方法加入 `Sequence`类。构造协议不变，只是补上 prompt 账本查询能力。

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

### 4.2 这组方法解决了什么

1. `Scheduler` 能直接问：这条序列还剩多少 prompt token 没算？
2. `ModelRunner` 能直接拿到：本轮该送进模型的那一段 chunk 是什么？
3. 代码和文档都不再把 `num_cached_tokens` 误读成”整条序列已经全算过多少 token”。

---

## 5. 修改 `Config`

在 `config.py` 的 `Config` 数据类里加一个参数：

```python
max_prefill_chunk_size: int = 1024
```

含义：单条序列在一次 prefill step 里最多处理多少未计算 prompt token。

它和 `max_num_batched_tokens` 是两回事：前者管单条序列每轮推进多少，后者管整个 batch 本轮总量。

---

## 6. 修改 `engine/scheduler.py`

### 6.1 调度器必须告诉 ModelRunner 本轮处理多少 token

如果 `schedule()` 仍然只返回 `list[Sequence]`，`ModelRunner` 就不知道每条 waiting 序列这轮究竟该处理多少 prompt token。所以把调度结果改成三元组：

```python
scheduled_seqs, is_prefill, prefill_chunk_sizes = scheduler.schedule()
```

- `scheduled_seqs`：本轮要处理的序列。
- `is_prefill`：是否走 prefill 路径。
- `prefill_chunk_sizes`：仅 prefill 时使用，每条序列这轮该处理多少 prompt token。

### 6.2 完整实现

替换当前 `Scheduler.schedule()`，并新增 `mark_prefill_progress()`：

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

一行，但少了它 chunk 大小就成了写死的隐藏常量。

---

## 7. 修改 `engine/model_runner.py`

### 7.1 `prepare_prefill()` 只处理本轮 chunk

当前版本把整条 `seq.token_ids` 全部喂进去，和 chunked prefill 冲突。替换成下面这版：

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

### 7.2 `run()` 接收 `chunk_sizes`

替换 `ModelRunner.run()`。这版假设 Day4 已经把 `run_model()` 拆出来，Day3 已经让 sampler 支持 `top_k / top_p`：

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

### 7.3 最容易看漏的点：positions

第二个 chunk 的 positions 绝不能从 0 重新开始。

一条 prompt 已经 prefill 了前 512 个 token，本轮再处理下一个 256-token chunk，positions 应该是：

```python
list(range(512, 768))
```

而不是：

```python
list(range(256))
```

写错这里，RoPE 和 KV 写入位置会一起错。

---

## 8. 修改 `engine/llm_engine.py`

`LLMEngine.step()` 要显式传递 chunk 信息。替换 `step()`：

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

### 8.1 prefill 后为什么还要走 `postprocess()`

即使这轮只处理了 prompt chunk，模型最后仍然会输出每条序列当前 step 的 logits，sampler 仍然会采出一个”下一 token”。所以这轮结束时两件事不能颠倒：

1. 先记账：本轮 chunk 对应的 prompt token 已进 cache（`mark_prefill_progress`）。
2. 再 append：本轮新生成的 decode token（`postprocess`）。

---

## 9. 新增 `tests/test_Day8_chunked_prefill.py`

测试目标不是跑大模型，而是锁住 chunk 账本和调度边界。下面这份可以直接用：

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

如果前面主线 Day0-7 已经跑通，再做一轮轻量手测：

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

预期：还剩 6个未计算 prompt token，本轮 chunk 取到索引 4、5、6 对应的 token。

---

## 11. 常见坑

1. **调度器切了 chunk，`prepare_prefill()` 还是把整条 prompt 全送进模型。** 最经典的”文义上支持 chunk，实际仍是全量 prefill”假实现。
2. **第二个 chunk 的 positions 又从 0 开始。** RoPE 和 cache slot 位置语义一起崩。
3. **把 `num_cached_tokens` 理解成”整条序列缓存了多少”，而不是”prompt 前缀里做完 prefill 的部分”。** decode 账本会越来越乱。
4. **prefill 完成后不把序列从 waiting 转到 running。** decode 永远接不上。
5. **一上来就重写 BlockManager。** 完全没必要。先把调度和输入窗口改正确。

---

## 12. 读完你应该明白

chunked prefill 本质上是在改”本轮 token budget 怎么分配”。当前仓库里最自然的接入点是 `Sequence → Scheduler → ModelRunner → LLMEngine.step()`。`positions` 和 `slot_mapping` 必须按 chunk 正确推进，否则就是假实现。

下一篇：`Day9-Radix-Prefix-Cache与可观测指标.md`——把 hash 表 prefix cache 升级成 prefix tree。

---

## 13. 文件级修改清单

| 文件 | 要写什么 | 别写什么 |
|---|---|---|
| `engine/sequence.py` | 补 `num_uncomputed_tokens / prefill_done / get_chunk_token_ids()` 等 prompt 账本接口 | 别把 `token_ids` 切成多个新请求对象，别把 decode token 算进未 prefill prompt |
| `config.py` | 新增 `max_prefill_chunk_size`，单条序列每轮 prefill 推进量可配置 | 别用隐藏常量，别让它覆盖 `max_num_batched_tokens` 的 batch 预算含义 |
| `engine/scheduler.py` | `schedule()` 返回 `prefill_chunk_sizes`，新增 prefill 进度记账接口 | 别继续要求长 prompt 一次性全部上车，别在调度器里跑模型 |
| `engine/model_runner.py` | `prepare_prefill()` 只准备本轮 chunk，按 `num_cached_tokens` 生成 positions 和 `slot_mapping` | 别把整条 prompt 重新送进模型，别让第二个 chunk 的 positions 从 0 开始 |
| `engine/llm_engine.py` | `step()` 传递 chunk 信息，prefill 吞吐只统计 `sum(chunk_sizes)` | 别用旧的整段 prefill 统计口径，别颠倒 prefill 记账和 append token 顺序 |
| `tests/test_Day8_chunked_prefill.py` | 轻量测试：Sequence 账本、scheduler chunk 返回、进度迁移、step token 统计 | 别加载真实模型，别写成 GPU 性能验证 |
