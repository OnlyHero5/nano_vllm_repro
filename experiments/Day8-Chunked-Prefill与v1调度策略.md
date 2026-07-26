# Day 8 — Chunked Prefill：别让长 prompt 堵死整个 batch

> **前置依赖**：本篇代码建立在 Day1-Day6 主线全部落地之后：
>
> - **Day4**：`Qwen3ForCausalLM.forward()` 只返回 hidden_states，`compute_logits()` 单独投影；
> - **Day5**：`ModelRunner` 已拆出 `run_model()`，`run()` 用 try/finally 包裹 `reset_context()`；
> - **Day6**：`Sequence` 已有 `top_k / top_p`，`Sampler.forward(logits, temperatures, top_ks, top_ps)` 已支持四参。
>
> 如果你还停留在仓库基线状态（`run()` 直接 `self.model(...)`、Sampler 只收 temperatures），先回 Day4/Day5/Day6 把主线补齐，否则本篇代码抄不进去。
>
> 另外本篇会用到 `flash_attn_varlen_func` 的 `block_table` 参数（分页 KV 读取），需要 **flash-attn >= 2.5**，且 KV cache 块大小是 256 的倍数（本仓库 `block_size=256`，`Config` 已强制校验）。

想象这个场景：waiting 队列头部排着一条 8000 token 的长 prompt，后面跟着三条十几个 token 的短请求。当前调度器的做法是——8000 token 塞不下？那谁也别上车。三条短请求继续干等。

这不太公平。

chunked prefill 的思路很直白：**长 prompt 不必一次全算完，拆成几段，每轮只推进一段。** 这轮算 1024 个 token，下轮再算 1024 个，省下来的 budget 留给别的请求。

这次改的是调度、输入准备和 attention 的 prefill 路径——模型主干、Triton kernel、decode 路径都不碰。

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
3. **prefix cache 命中也没用。** 虽然 `num_cached_tokens` 会减小 `new_tokens`，但调度器仍然是"整条剩余 prompt 一次性尝试塞入"。

chunked prefill 的改法：**允许一条 prefill 请求这轮只处理一部分 token，下轮接着来。** 这和 vLLM v1 的调度思路一致——token budget 是本轮的核心资源，prefill 可以被切成多个 step，decode 不再总被长 prompt 挤压。

---

## 2. 与本篇相关的五处代码

### 2.1 `Sequence`：有总账，没细账

当前 `Sequence` 已经有 `token_ids`、`num_prompt_tokens`、`num_cached_tokens`、`block_table`——能表达"一共多少 token，多少已进 cache"。但缺一个关键问题：**本轮该处理到哪？**

### 2.2 `Scheduler.schedule()`：整条 prompt 一次性尝试装入

waiting 路径是：取队首 → 算 `new_tokens` → 塞不下就 `break`。队头那条长请求一旦塞不下，后面更短的请求全部被卡住。

### 2.3 `ModelRunner.prepare_prefill()`：总是送整条 `seq.token_ids`

```python
token_ids = seq.token_ids
seq_len = len(token_ids)
all_token_ids.extend(token_ids)
all_positions.extend(range(seq_len))
```

这是"全量 prefill"，不是"只 prefill 还没算过的那一段"。

### 2.4 `layers/attention.py`：prefill 路径只会看"本轮送进来的 K/V"

`_prefill_attention()` 把当前 batch 的 k、v 直接喂给 `flash_attn_varlen_func`，`cu_seqlens_k` 就是本轮 token 的边界。第一轮 chunk 没问题；**第二轮 chunk 时前缀的 K/V 只存在 KV cache 里，不在本轮的 k、v 张量里**——这条路径必须学会从分页 cache 读前缀（详见 §3.2）。

### 2.5 `LLMEngine.step()`：吞吐统计也按整条 prefill 在算

所以这篇必须同时改调度、输入准备和 attention。只改一头就会出现经典翻车：调度器以为自己切了 chunk，`ModelRunner` 还是把整条 prompt 重新送进模型；或者更隐蔽的——chunk 切了，但 attention 根本看不到前缀。

---

## 3. chunked prefill 的账本与数学

### 3.1 账本：两个数字

不新增复杂请求对象，只在 `Sequence` 上补齐一组运行时语义：

1. `num_tokens`：当前序列总长（prompt + 已生成 token）。
2. `num_cached_tokens`：已进 KV cache、不需要再计算的前缀长度。

```python
num_uncomputed_tokens = seq.num_tokens - seq.num_cached_tokens
```

注意这里按**整条序列**记账，而不是只看 prompt 段。原因：被抢占（preempt）的序列会带着已生成的 token 回到 waiting 队列重新 prefill（`num_cached_tokens` 被清零），这时需要重算的是"prompt + 已生成 token"整段。按 `num_tokens` 记账，普通新请求和抢占重算走的是同一套逻辑。

### 3.2 数学：第二个 chunk 的 attention 必须看到前缀

这是本篇**最容易写错、错了还不容易发现**的地方。

设一条 prompt 长 768，第一轮算了 [0, 512)，第二轮算 [512, 768)。第二轮里位置 512 的 query 在数学上必须对 **key [0..512]** 做注意力——包括第一轮已经算完、存进 KV cache 的那 512 个 token。

如果第二轮只把本轮 chunk 的 k、v 喂给 `flash_attn_varlen_func`，并把 `cu_seqlens_k` 设成 chunk 长度，那么位置 512 的 query 只能看到位置 [512..768) 的 key——**注意力结果在数学上是错的**（模型输出会直接坏掉，而且不报任何错）。

正确做法（与上游 nano-vllm 的 prefix cache 路径一致）：

1. `store_kvcache` 先把本轮 chunk 的 K/V 写入分页 cache（这一步现有代码已经会做，slot_mapping 指向 chunk 的槽位）。此时 cache 里已连续存有每条序列 [0, end) 的全部 K/V。
2. attention 不再用本轮的 k、v 张量，而是把**分页 cache 本身**连同 `block_table` 传给 `flash_attn_varlen_func`：

```python
flash_attn_varlen_func(
    q=q,                       # 只有本轮 chunk 的 query
    k=k_cache,                 # [num_blocks, block_size, num_kv_heads, head_dim]
    v=v_cache,                 # 同上——分页 cache 本身，不是本轮的 k/v 张量
    cu_seqlens_q=cu_seqlens_q,  # 按 chunk 长度累积
    cu_seqlens_k=cu_seqlens_k,  # 按「前缀 + chunk」总长累积 ← 关键
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=scale,
    causal=True,
    block_table=block_tables,  # 每条序列的逻辑块 → 物理页映射
)
```

3. `causal=True` 时 flash-attn 把因果掩码对齐到注意力矩阵的**右下角**（bottom-right alignment，flash-attn ≥ 2.1 的语义）：chunk 内第 j 个 query 能看到 key `[0 .. seqlen_k - seqlen_q + j]`。代入 `seqlen_k = cached + chunk_len`、`seqlen_q = chunk_len`，正好是全局位置 `[0 .. cached + j]`——与"整条序列一次算完"的因果语义完全一致。

### 3.3 不是拆序列，是拆本轮输入窗口

第一次看 chunked prefill 容易想歪：是不是要把 `token_ids` 切成几个子序列对象？

不用。`token_ids` 继续保存完整序列，`num_cached_tokens` 继续标记"已进 cache 的前缀"，每轮由调度器决定这次最多再前进多少 token。chunked prefill 是给现有序列指定一个本轮处理窗口，不是重新发明序列对象。

---

## 4. 修改 `engine/sequence.py`

### 4.1 增加 chunked prefill 辅助属性

把下面这组方法加入 `Sequence` 类。构造协议不变，只是补上账本查询能力。

```python
@property
def num_uncomputed_tokens(self) -> int:
    """
    整条序列里还没算进 KV cache 的 token 数。

    覆盖两种情况：
    - 新请求：prompt 还没算完的部分。
    - 抢占重算：num_cached_tokens 被清零后，prompt + 已生成 token 整段都要重算。
    """
    return max(0, self.num_tokens - self.num_cached_tokens)


@property
def prefill_done(self) -> bool:
    """当前已知上下文是否已经全部进入 KV cache。"""
    return self.num_cached_tokens >= self.num_tokens


def get_chunk_token_ids(self, chunk_size: int) -> list[int]:
    """
    返回本轮需要执行 prefill 的那一段 token。

    约定：
    - chunk 一定从 num_cached_tokens 开始。
    - 如果剩余不足 chunk_size，就返回剩余全部。
    """
    assert chunk_size > 0, "chunk_size 必须 > 0"
    start = self.num_cached_tokens
    end = min(self.num_tokens, start + chunk_size)
    return self.token_ids[start:end]
```

### 4.2 这组方法解决了什么

1. `Scheduler` 能直接问：这条序列还剩多少 token 没算？
2. `ModelRunner` 能直接拿到：本轮该送进模型的那一段 chunk 是什么？
3. 抢占重算和普通新请求共用同一套账本，不需要特判。

---

## 5. 修改 `config.py`

在 `config.py` 的 `Config` 数据类里加一个参数：

```python
max_prefill_chunk_size: int = 1024
```

含义：单条序列在一次 prefill step 里最多处理多少未计算 token。

它和 `max_num_batched_tokens` 是两回事：前者管单条序列每轮推进多少，后者管整个 batch 本轮总量。

---

## 6. 修改 `engine/scheduler.py`

### 6.1 调度器必须告诉 ModelRunner 本轮处理多少 token

如果 `schedule()` 仍然只返回 `(seqs, is_prefill)`，`ModelRunner` 就不知道每条 waiting 序列这轮究竟该处理多少 token。所以把调度结果改成三元组：

```python
scheduled_seqs, is_prefill, prefill_chunk_sizes = scheduler.schedule()
```

- `scheduled_seqs`：本轮要处理的序列。
- `is_prefill`：是否走 prefill 路径。
- `prefill_chunk_sizes`：仅 prefill 时使用，每条序列这轮该处理多少 token。

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

    # ===== 阶段 1：chunked prefill =====
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        if seq in scheduled_seqs:
            # 部分 chunk 的序列会被放回队尾；再次在队首看到它，
            # 说明本轮已经绕队列一圈。同一 batch 里绝不能出现
            # 同一序列的两个 chunk（num_cached_tokens 还没更新，
            # 两个 chunk 会算成同一段）。
            break

        # 首次调度时为整条序列分配 blocks
        if not seq.block_table:
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            # 注意：allocate() 内部 prefix cache 命中会推进 num_cached_tokens，
            # 所以 chunk 的起点必须在 allocate 之后再读。

        remaining_tokens = seq.num_uncomputed_tokens
        if remaining_tokens <= 0:
            # 比如 prefix cache 全命中：无需计算，直接转 running
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            continue

        remaining_budget = self.max_num_batched_tokens - num_batched_tokens
        chunk_size = min(remaining_tokens, self.max_prefill_chunk_size, remaining_budget)
        if chunk_size <= 0:
            break

        self.waiting.popleft()
        scheduled_seqs.append(seq)
        prefill_chunk_sizes.append(chunk_size)
        num_seqs += 1
        num_batched_tokens += chunk_size

        # 本轮不会完成 prefill 的序列放回队尾，让后续短请求也有机会被调度。
        # 完成的序列先留在 scheduled 列表里，由 mark_prefill_progress 转入 running。
        if chunk_size < remaining_tokens:
            self.waiting.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True, prefill_chunk_sizes

    # ===== 阶段 2：decode（与当前主线一致）=====
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
    在一轮 prefill 完成后，把本轮已经算完并写入 cache 的 token 记到账本里。
    """
    assert len(seqs) == len(chunk_sizes), "seqs 与 chunk_sizes 必须一一对应"

    for seq, chunk_size in zip(seqs, chunk_sizes):
        seq.num_cached_tokens = min(seq.num_tokens, seq.num_cached_tokens + chunk_size)

        if seq.prefill_done:
            seq.status = SequenceStatus.RUNNING
            if seq not in self.running:
                self.running.append(seq)
        else:
            seq.status = SequenceStatus.WAITING
            if seq not in self.waiting:
                self.waiting.append(seq)
```

注意 preempt 路径不用改：`__preempt()` 调 `block_manager.deallocate(seq)`，它会把 `num_cached_tokens` 清零——被抢占的序列回到 waiting 后自动从头开始重算，chunk 账本天然覆盖这种情况。

### 6.3 在 `Scheduler.__init__()` 里接入参数

```python
self.max_prefill_chunk_size = getattr(config, "max_prefill_chunk_size", 1024)
```

一行，但少了它 chunk 大小就成了写死的隐藏常量。

---

## 7. 修改 `engine/model_runner.py`

### 7.1 `prepare_prefill()` 只处理本轮 chunk，K 覆盖整个前缀

当前版本把整条 `seq.token_ids` 全部喂进去，和 chunked prefill 冲突。替换成下面这版。三个关键变化：

1. 每条序列这轮只处理一个 chunk，`positions` 从 `num_cached_tokens` 开始。
2. `cu_seqlens_q` 按 chunk 长度累积，**`cu_seqlens_k` 按「前缀 + chunk」总长累积**——这就是 §3.2 说的"attention 必须看到前缀"。
3. 只要 batch 里有任何序列带前缀，就把 `block_tables` 塞进 Context，让 attention 走分页 cache 读取路径。

```python
def prepare_prefill(
    self,
    sequences: list[Sequence],
    chunk_sizes: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """准备 chunked prefill 阶段的输入。"""
    assert len(sequences) == len(chunk_sizes), "sequences 与 chunk_sizes 必须一一对应"

    all_token_ids: list[int] = []
    all_positions: list[int] = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping: list[int] = []
    has_prefix = False

    for seq, chunk_size in zip(sequences, chunk_sizes):
        chunk_token_ids = seq.get_chunk_token_ids(chunk_size)
        start = seq.num_cached_tokens
        end = start + len(chunk_token_ids)

        # query 只覆盖本轮 chunk；positions 从全局位置 start 开始（RoPE 依赖它）
        all_token_ids.extend(chunk_token_ids)
        all_positions.extend(range(start, end))

        seqlen_q = len(chunk_token_ids)
        seqlen_k = end  # 前缀 + 本轮 chunk：K 必须覆盖整个已知上下文
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, seqlen_k)
        if start > 0:
            has_prefix = True

        # slot_mapping 只覆盖本轮新写入 cache 的 chunk 段
        for pos in range(start, end):
            block_id = seq.block_table[pos // self.block_size]
            slot_mapping.append(block_id * self.block_size + pos % self.block_size)

    # 任何序列带前缀时，attention 需要按 block_table 从分页 cache 读 K/V
    block_tables = None
    if has_prefix:
        max_num_blocks = max(len(seq.block_table) for seq in sequences)
        padded = [
            seq.block_table + [0] * (max_num_blocks - len(seq.block_table))
            for seq in sequences
        ]
        block_tables = torch.tensor(padded, dtype=torch.int32, device=self.device)

    input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=self.device)
    positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)

    context = Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device),
        cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
        context_lens=None,
        block_tables=block_tables,  # 非 None 时触发 attention 的分页读取路径
        max_context_len=None,
        max_num_blocks=None,
        kv_cache=self.kv_cache,
    )
    set_context(context)
    return input_ids, positions
```

说明两点：

- 没有前缀时（batch 里全是第一个 chunk），`cu_seqlens_k` 与 `cu_seqlens_q` 相等、`block_tables` 为 `None`，attention 走原来的普通 varlen 路径——旧行为完全保留。
- `block_tables` 的 padding 值用 0：flash-attn 只会按 `cu_seqlens_k` 声明的长度读块，padding 部分永远不会被访问。

### 7.2 `run()` 接收 `chunk_sizes`

替换 `ModelRunner.run()`。这版用的是 Day5 拆出的 `run_model()`（内部已按 `cu_seqlens_q` 取每条序列 chunk 最后一个位置的 logits）和 Day6 的四参 Sampler：

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
            [seq.top_k for seq in sequences],
            dtype=torch.int32,
            device=self.device,
        )
        top_ps = torch.tensor(
            [seq.top_p for seq in sequences],
            dtype=torch.float32,
            device=self.device,
        )
        next_tokens = self.sampler(logits, temperatures, top_ks, top_ps)
        return next_tokens.tolist()
    finally:
        reset_context()
```

Day5 的 `run_model()` 里那句 `last_token_indices = context.cu_seqlens_q[1:] - 1` 在 chunked prefill 下依然正确——它取的是**本轮 chunk** 的最后一个位置。这个位置的 logits 对"部分 chunk"没用（见 §9.1），但对"最后一个 chunk"正是我们要的 next-token 预测。

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

## 8. 修改 `layers/attention.py`

### 8.1 `_prefill_attention()` 增加分页前缀路径

这是 §3.2 数学的落地。替换 `Attention._prefill_attention()`：

```python
def _prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
) -> torch.Tensor:
    """Prefill: flash_attn_varlen_func

    两条子路径：
    - block_tables 为 None：普通 prefill，K/V 就是本轮 batch 的 k、v。
    - block_tables 非 None：chunked prefill 的后续 chunk（或 prefix cache 命中），
      本轮 chunk 的 K/V 已由 store_kvcache 写入分页 cache，
      前缀的 K/V 也在 cache 里——直接把分页 cache 交给 flash-attn，
      用 block_table 按页读取，cu_seqlens_k 覆盖「前缀 + chunk」总长。
    """
    original_dtype = q.dtype
    block_tables = context.block_tables

    if block_tables is not None:
        kv_cache = context.kv_cache[self.layer_idx]
        k = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]，已是 fp16
        v = kv_cache[1]
    else:
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    output = flash_attn_varlen_func(
        q=q.to(torch.float16),
        k=k,
        v=v,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        softmax_scale=self.scale,
        causal=True,
        block_table=block_tables,
    )
    return output.to(original_dtype)
```

要点：

- `store_kvcache` 在 `forward()` 里先于 attention 执行（现有代码顺序不变），所以走到这里时 cache 已包含每条序列 `[0, end)` 的完整 K/V。
- `causal=True` 在 `seqlen_q < seqlen_k` 时按右下角对齐掩码，chunk 内每个 query 恰好能看到自己的全局因果范围（§3.2 推导过）。
- `block_table` 参数要求 flash-attn >= 2.5，分页块大小是 256 的倍数——本仓库 `block_size=256` 满足。
- decode 路径（`flash_attn_with_kvcache`）一行不用改。

---

## 9. 修改 `engine/llm_engine.py`

### 9.1 部分 chunk 的采样结果必须丢弃

模型每轮都会为每条序列的 chunk 末位采出一个 token，但它是否有意义要分情况：

| 本轮结束时序列状态 | chunk 末位 logits 预测的是 | 采样结果怎么处理 |
|---|---|---|
| chunk 未覆盖完（部分 chunk） | 下一个**已知**的 token（它就在 `token_ids` 里） | **丢弃**。把它 append 进序列会污染上下文 |
| 抢占重算完成（已有生成 token） | 最后一个已生成 token 的下一个 | 丢弃，留给下一轮 decode 采（块账本只在 decode 调度里推进）|
| 新 prompt 本轮全部算完 | 真正的 next token | append 进序列（和原来的 prefill 语义一致） |

### 9.2 替换 `step()`

```python
def step(self) -> tuple[list[tuple[int, list[int]]], int]:
    """
    执行一次调度和推理。

    num_tokens 约定：
    - prefill 阶段返回正数，表示本轮新计算的 token 数（只算 chunk）。
    - decode 阶段返回负数，绝对值表示本轮处理了多少条序列。
    """
    seqs, is_prefill, chunk_sizes = self.scheduler.schedule()
    if not seqs:
        return [], 0

    if is_prefill:
        num_tokens = sum(chunk_sizes)
        token_ids = self.model_runner.run(seqs, True, chunk_sizes)

        # 1) 先记账：本轮 chunk 已进 KV cache，完成 prefill 的序列转入 running
        self.scheduler.mark_prefill_progress(seqs, chunk_sizes)

        # 2) 再决定谁的采样结果可以写回（见 §9.1 的表）
        sampled_seqs: list[Sequence] = []
        sampled_tokens: list[int] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.prefill_done and seq.num_completion_tokens == 0:
                sampled_seqs.append(seq)
                sampled_tokens.append(token_id)
        finished_seqs = self.scheduler.postprocess(sampled_seqs, sampled_tokens)
    else:
        num_tokens = -len(seqs)
        token_ids = self.model_runner.run(seqs, False)
        finished_seqs = self.scheduler.postprocess(seqs, token_ids)

    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
    return outputs, num_tokens
```

记账（`mark_prefill_progress`）和写回（`postprocess`）的顺序不能颠倒：`prefill_done` 的判断依赖记账后的 `num_cached_tokens`。

---

## 10. 新增 `tests/test_Day8_chunked_prefill.py`

测试目标不是跑大模型，而是锁住 chunk 账本和调度边界（`Config` 构造需要本地 `models/Qwen3-0.6B` 目录存在，但不加载权重）：

```python
"""Day8 chunked prefill 结构测试。"""

import sys

sys.path.insert(0, ".")

from sampling_params import SamplingParams
from engine.sequence import Sequence, SequenceStatus


def test_sequence_chunk_properties():
    seq = Sequence([10, 11, 12, 13, 14, 15], SamplingParams())

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
    # 两条都是部分 chunk，应回到 waiting 队列等待下一轮
    assert seq1 in scheduler.waiting and seq2 in scheduler.waiting


def test_mark_prefill_progress_moves_finished_to_running():
    from config import Config
    from engine.block_manager import BlockManager
    from engine.scheduler import Scheduler

    config = Config(model_path="models/Qwen3-0.6B")
    block_manager = BlockManager(num_blocks=16, block_size=256)
    scheduler = Scheduler(config, block_manager)

    seq = Sequence([1, 2, 3], SamplingParams())
    scheduler.mark_prefill_progress([seq], [3])

    assert seq.num_cached_tokens == 3
    assert seq.prefill_done is True
    assert seq.status == SequenceStatus.RUNNING
    assert seq in scheduler.running


def test_partial_chunk_discards_sampled_token():
    from engine.llm_engine import LLMEngine

    seq = Sequence([1, 2, 3, 4, 5, 6], SamplingParams())

    class FakeScheduler:
        def schedule(self):
            return [seq], True, [2]

        def mark_prefill_progress(self, seqs, chunk_sizes):
            assert seqs == [seq] and chunk_sizes == [2]
            seq.num_cached_tokens += 2

        def postprocess(self, seqs, token_ids):
            # 部分 chunk：不应有任何序列进入 postprocess
            assert seqs == [] and token_ids == []
            return []

    class FakeRunner:
        def run(self, seqs, is_prefill, chunk_sizes=None):
            assert is_prefill is True and chunk_sizes == [2]
            return [99]

    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = FakeScheduler()
    engine.model_runner = FakeRunner()

    outputs, num_tokens = engine.step()

    assert num_tokens == 2
    assert outputs == []
    assert seq.num_tokens == 6  # 采样出的 99 被丢弃，没有写进序列


def test_final_chunk_appends_sampled_token():
    from engine.llm_engine import LLMEngine

    seq = Sequence([1, 2, 3, 4], SamplingParams())
    seq.num_cached_tokens = 2  # 前一轮已算完 2 个 token

    class FakeScheduler:
        def schedule(self):
            return [seq], True, [2]

        def mark_prefill_progress(self, seqs, chunk_sizes):
            seq.num_cached_tokens += 2
            seq.status = SequenceStatus.RUNNING

        def postprocess(self, seqs, token_ids):
            assert seqs == [seq] and token_ids == [99]
            seq.append_token(99)
            return []

    class FakeRunner:
        def run(self, seqs, is_prefill, chunk_sizes=None):
            return [99]

    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = FakeScheduler()
    engine.model_runner = FakeRunner()

    outputs, num_tokens = engine.step()

    assert num_tokens == 2
    assert seq.token_ids[-1] == 99


if __name__ == "__main__":
    test_sequence_chunk_properties()
    test_scheduler_returns_chunk_sizes()
    test_mark_prefill_progress_moves_finished_to_running()
    test_partial_chunk_discards_sampled_token()
    test_final_chunk_appends_sampled_token()
    print("Day8 chunked prefill tests passed")
```

---

## 11. 验收命令

```bash
python -m py_compile engine/sequence.py engine/scheduler.py engine/model_runner.py engine/llm_engine.py layers/attention.py tests/test_Day8_chunked_prefill.py
python tests/test_Day8_chunked_prefill.py
```

如果前面主线 Day0-7 已经跑通，再做一轮端到端手测（把 `max_prefill_chunk_size` 调小，逼出多 chunk 路径）：

```bash
python - <<'PY'
from llm import LLM
from sampling_params import SamplingParams

llm = LLM("models/Qwen3-0.6B", max_prefill_chunk_size=64)
out = llm.generate(
    ["请把下面这句话翻译成英文：今天天气很好。" * 8],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(out[0]["text"])
PY
```

验收标准：输出与 `max_prefill_chunk_size=1024`（一轮算完）时**完全一致**（greedy 下逐 token 相同）。如果第二个 chunk 的 attention 看不到前缀，这里的输出会明显劣化甚至乱码——这正是锁正确性的黄金测试。

---

## 12. 常见坑

1. **第二个 chunk 只传本轮的 `cu_seqlens_k`，attention 看不到已缓存前缀。** 本篇最致命的坑：不报错，但注意力在数学上是错的。`cu_seqlens_k` 必须覆盖「前缀 + chunk」总长，且带前缀时必须走分页 cache + `block_table` 路径（§3.2 / §8.1）。
2. **第二个 chunk 的 positions 又从 0 开始。** RoPE 和 cache slot 位置语义一起崩。
3. **部分 chunk 的采样结果被 append 进序列。** chunk 末位 logits 预测的是下一个已知 prompt token，写回会污染上下文（§9.1）。
4. **调度器切了 chunk，`prepare_prefill()` 还是把整条 prompt 全送进模型。** 最经典的"文义上支持 chunk，实际仍是全量 prefill"假实现。
5. **同一 batch 里调度了同一序列的两个 chunk。** 第二个 chunk 的起点还是旧的 `num_cached_tokens`，两个 chunk 会重叠。`schedule()` 里的 `seq in scheduled_seqs` 检查就是防这个。
6. **prefill 完成后不把序列从 waiting 转到 running。** decode 永远接不上。
7. **一上来就重写 BlockManager。** 完全没必要。blocks 在首次调度时一次性按整条序列分配，chunk 只影响"每轮算多少"，不影响"占多少显存"。

---

## 13. 读完你应该明白

chunked prefill 本质上是在改"本轮 token budget 怎么分配"，但它牵一发动全身：`Sequence` 的账本、`Scheduler` 的三元组返回、`prepare_prefill` 的窗口切割、attention 的分页前缀读取、`step()` 的采样写回过滤，五处必须一起对。其中数学上最关键的一条：**后续 chunk 的 K 必须覆盖整个已缓存前缀**，靠 `cu_seqlens_k` + `block_table` + flash-attn 的右下角因果对齐实现。

下一篇：`Day9-Radix-Prefix-Cache与可观测指标.md`——把 hash 表 prefix cache 升级成 prefix tree。

---

## 14. 文件级修改清单

| 文件 | 要写什么 | 别写什么 |
|---|---|---|
| `engine/sequence.py` | 补 `num_uncomputed_tokens / prefill_done / get_chunk_token_ids()` 账本接口（按 `num_tokens` 整条序列记账） | 别把 `token_ids` 切成多个新请求对象 |
| `config.py` | 新增 `max_prefill_chunk_size`，单条序列每轮 prefill 推进量可配置 | 别用隐藏常量，别让它覆盖 `max_num_batched_tokens` 的 batch 预算含义 |
| `engine/scheduler.py` | `schedule()` 返回 `prefill_chunk_sizes`，新增 `mark_prefill_progress()`，防同轮重复调度 | 别继续要求长 prompt 一次性全部上车，别在调度器里跑模型 |
| `engine/model_runner.py` | `prepare_prefill(sequences, chunk_sizes)`：chunk 窗口 + `cu_seqlens_k` 覆盖前缀 + 带前缀时设置 `block_tables` | 别把整条 prompt 重新送进模型，别让 `cu_seqlens_k` 只覆盖 chunk |
| `layers/attention.py` | `_prefill_attention()` 增加分页前缀路径：`block_tables` 非 None 时 K/V 取分页 cache，传 `block_table` 给 flash-attn | 别动 decode 路径和 store_kvcache kernel |
| `engine/llm_engine.py` | `step()` 传递 chunk 信息；先记账再写回；部分 chunk / 抢占重算的采样结果丢弃 | 别把部分 chunk 的采样 token append 进序列 |
| `tests/test_Day8_chunked_prefill.py` | 轻量测试：Sequence 账本、scheduler chunk 返回、进度迁移、采样写回过滤 | 别加载真实模型，别写成 GPU 性能验证 |
