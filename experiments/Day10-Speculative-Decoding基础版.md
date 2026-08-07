# Day 10 — Speculative Decoding：让小模型先猜，大模型来验

> **前置依赖**：
>
> - **Day4**：`Qwen3ForCausalLM.forward()` 返回 hidden_states，`compute_logits()` 单独投影——本篇的验证接口要拿"整段位置"的 logits，靠的就是这个拆分；
> - **Day8**：`Scheduler.schedule()` 已改为返回三元组 `(seqs, is_prefill, chunk_sizes)`，`LLMEngine.step()` 已是 Day8 版本——本篇 `step()` 的 speculative 分支直接改在它上面。
>
> （Day8 又依赖 Day5/Day6，所以实际顺序是：主线 Day1-6 → Day8 → 本篇。）

主线解决的是"怎么把一个模型高效跑起来"。但它默认每轮 decode 只由主模型自己预测一个 token。

speculative decoding 打破这个默认：

1. 先让一个更便宜的 draft 模型一次猜 `k` 个 token。
2. 再让主模型一次验证这串 draft token。
3. 能接受的整段接受，不能接受的回退到主模型真实输出。

主模型很贵、草稿模型便宜时，这能显著减少主模型的单 token decode 次数。对长生成来说，吞吐提升往往很明显。

关键认知：**speculative decoding 是一层新的生成控制流，不是新的 attention kernel。** 所以这次不碰 Attention、KV Cache 布局和 Scheduler 状态机——在 `Scheduler → ModelRunner → postprocess` 这条链路外面包一层 draft-verify 逻辑就够了。Medusa、tree-based verifier、TP/CUDA Graph 联动——都不做。

---

## 0. 先把话说清楚：教学版的两个取舍

动手前必须明白这版实现的两个（有意的）妥协，别把它当成生产实现：

**取舍一：这是 greedy 精确匹配验证，不是 Leviathan 接受采样。**

论文（Leviathan et al., 2023）里的 speculative decoding 用**接受采样**（acceptance sampling）：对每个 draft token $x$，以概率 $\min(1, p(x)/q(x))$ 接受（$p$ 是 target 分布，$q$ 是 draft 分布），拒绝时从修正分布 $\mathrm{norm}(\max(0, p-q))$ 重采。这保证输出分布**与 target 模型采样完全一致**。

本篇的教学版做的是更简单的事：target 用 greedy（argmax）给出它自己的 token 路径，draft token 与之逐位比较，第一处不一致就截断。这个方案：

- 对 **greedy 解码（temperature=0）是无损的**——输出与 target 自己 greedy 生成逐 token 相同；
- 对随机采样请求**不是**无损的（它把输出强行变成了 greedy），所以教学版只对 `temperature=0` 的请求开启。

§9 给出了升级到接受采样需要补什么。

**取舍二：draft 和 target 都不复用 KV cache。**

教学版每轮 draft 提议和 target 验证都对整段上下文做一次**不写 cache 的全量前向**。好处是控制流干净：不碰主线的块账本，不存在"验证用的临时序列泄漏 block"这类问题。代价是真实的：每轮的计算量是 $O(\text{上下文长度})$ 而不是 $O(1)$，**上下文一长，这版实现会比普通 decode 更慢**。它的价值是把 accept/reject 语义写对，性能优化（draft/target KV cache 复用）是之后的事。

---

## 1. 一轮 speculative decode 长什么样

对每条 running 序列：

1. draft model 基于当前上下文，连续提议 `k` 个 token。
2. target model 一次性验证这 `k` 个位置。
3. 从左到右比较：draft token 与 target token 一致就接受；第一次不一致就停。
4. 全部接受？再补一个 target model 决定的真实 next token。
5. 中间拒绝？保留已接受的前缀，把第一个拒绝位置改成 target token。

---

## 2. 与本篇相关的三处代码

### 2.1 `ModelRunner.run()` 只返回"每条序列一个 next token"

普通 decode 够用，但 speculative 的验证阶段需要一次处理"当前上下文 + 一段草稿 token"，得到整段位置上的 logits。Day4 的 `forward()/compute_logits()` 拆分正好给了我们这个能力。

### 2.2 `Sampler` 职责很纯：从 logits 采样

别把 accept/reject 塞进 `Sampler`。教学版的验证是确定性的 argmax，连 Sampler 都不需要。

### 2.3 `Sequence` 已经够用

不需要重写。教学版只需要在当前 step 内把"最终接受的 token 列表"逐个追加回 `Sequence`——但注意块账本要跟上（§7.2）。

---

## 3. 两个模型，两个角色

- **target model**：当前仓库已有的主模型，结果以它为准。
- **draft model**：更小、更便宜，只负责提议 token。

> **隐含约束**：draft和 target 必须共享同一个 tokenizer（词表）。accept/reject 逐 token 比较的是 token ID，词表不同则比较无意义。本篇端到端验证用同一个模型同时充当 draft 和 target，天然满足；换真实小模型做 draft 时，确认它和 target 用同一份 tokenizer。

---

## 4. 修改 `config.py`

在 `Config` 里加一组显式开关，让这条路径不会默默影响主线：

```python
# ===== speculative decoding =====
enable_speculative_decoding: bool = False
draft_model_path: str | None = None
speculative_k: int = 4
```

在 `__post_init__()` 里加校验：

```python
if self.enable_speculative_decoding:
    assert self.draft_model_path is not None, "开启 speculative decoding 时必须提供 draft_model_path"
    assert os.path.isdir(self.draft_model_path), f"草稿模型路径不存在：{self.draft_model_path}"
    assert self.speculative_k > 0, "speculative_k 必须 > 0"
```

---

## 5. 修改 `engine/model_runner.py`：不写 cache 的验证前向

新增三个方法到 `ModelRunner`。核心是 `forward_logits_no_cache()`：对一段完整上下文做一次 prefill 式前向，但 Context 里 `kv_cache=None`、`slot_mapping=None`——`Attention.forward()` 会因此跳过 `store_kvcache`，走普通 varlen 路径。**不需要给临时序列分配 block，也就不存在 block 泄漏。**

> **import 提示**：下面的代码用到了 `reset_context()`，但基线 `model_runner.py:28` 只导入了 `Context, set_context, get_context`。在文件顶部的 import 行补上 `reset_context`：`from utils.context import Context, set_context, get_context, reset_context`。

```python
@torch.inference_mode()
def forward_logits_no_cache(self, token_ids: list[int]) -> torch.Tensor:
    """
    对一段完整上下文做一次不读写 KV cache 的前向，返回所有位置的 logits。

    与 run_model() 的区别：
    - run_model() 在 prefill 时只保留每条序列最后一个位置的 logits；
      这里需要整段位置（验证要逐位比较），所以直接调 model + compute_logits。
    - Context 里 kv_cache / slot_mapping 都是 None：
      Attention 层会跳过 store_kvcache，flash_attn_varlen_func 只对
      本轮张量做注意力——不碰任何分页 cache，不需要 block_table。
    """
    n = len(token_ids)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
    positions = torch.arange(n, dtype=torch.long, device=self.device)

    cu_seqlens = torch.tensor([0, n], dtype=torch.int32, device=self.device)
    set_context(Context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=n,
        max_seqlen_k=n,
        slot_mapping=None,
        kv_cache=None,
    ))
    try:
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)  # [n, vocab_size]
    finally:
        reset_context()


@torch.inference_mode()
def decode_one_token(self, token_ids: list[int]) -> int:
    """给定一条完整上下文，greedy 返回下一个 token（草稿模型用）。"""
    logits = self.forward_logits_no_cache(token_ids)
    return int(logits[-1].argmax(dim=-1).item())


@torch.inference_mode()
def verify_token_sequence(
    self,
    prefix_token_ids: list[int],
    draft_tokens: list[int],
) -> list[int]:
    """
    target model 验证一段 draft token。

    输出长度为 len(draft_tokens) + 1：
    - 位置 i（i < len(draft_tokens)）是 target 在"prefix + 前 i 个 draft token"
      之后 greedy 会选的 token——与 draft_tokens[i] 比较用。
    - 最后 1 个位置是 draft 全接受时 target 额外给出的 next token。
    """
    full_tokens = list(prefix_token_ids) + list(draft_tokens)
    logits = self.forward_logits_no_cache(full_tokens)

    # 位置 p 的 logits 预测 token p+1，所以从 len(prefix)-1 开始切
    start = len(prefix_token_ids) - 1
    return logits[start:].argmax(dim=-1).tolist()
```

### 5.1 为什么 verify 用 greedy（argmax）

target model 的验证阶段，本质上是在判断"它自己真正愿意走哪条 token 路径"。教学版用确定性 argmax，accept/reject 语义才稳定、可测试。这也正是 §0 说的取舍一：整条链路等价于 target 的 greedy 解码。

---

## 6. 新增 `engine/speculative.py`

别把所有 speculative 逻辑塞进 `LLMEngine` 或 `ModelRunner`。新建一个文件，专门负责 draft 提议、target 验证、accept/reject 计算：

```python
"""教学版 speculative decoding。

这个文件只做控制流，不碰 Attention、KV cache tensor 布局和调度器内部队列。
"""

from dataclasses import dataclass

import torch

from engine.sequence import Sequence


@dataclass
class SpeculativeResult:
    """
    一轮 speculative decode 的结果。

    accepted_tokens:
        经过 target model 验证后，最终可以真正写回 Sequence 的 token 列表。
    num_accepted_draft_tokens:
        这一轮有多少个 draft token 被接受。
    rejected:
        是否发生过拒绝。
    """
    accepted_tokens: list[int]
    num_accepted_draft_tokens: int
    rejected: bool


class DraftModelRunner:
    """
    草稿模型运行器。

    只依赖一个接口：decode_one_token(token_ids) -> int。
    传入的是草稿模型的 ModelRunner（§8 装配）。
    """

    def __init__(self, draft_model_runner):
        self.runner = draft_model_runner

    @torch.inference_mode()
    def propose_tokens(self, seq: Sequence, k: int) -> list[int]:
        """
        基于当前 Sequence 提议 k 个草稿 token。

        教学版直接循环调用草稿模型的单步接口（每步一次全量前向，
        不复用 KV cache——见 Day10 §0 取舍二）。
        """
        assert k > 0, "k 必须 > 0"

        proposed: list[int] = []
        working_tokens = list(seq.token_ids)

        for _ in range(k):
            next_token = self.runner.decode_one_token(working_tokens)
            proposed.append(next_token)
            working_tokens.append(next_token)

        return proposed


class SpeculativeVerifier:
    """
    speculative decoding 的接受 / 拒绝逻辑（greedy 精确匹配版）。
    """

    def __init__(self, target_model_runner):
        self.target = target_model_runner

    @torch.inference_mode()
    def verify(self, seq: Sequence, draft_tokens: list[int]) -> SpeculativeResult:
        """
        验证一段 draft token。

        规则：
        1. target model 一次性对 [context + draft_tokens] 做前向。
        2. 从左到右比较 target token 与 draft token。
        3. 第一次不一致：截断，接受的前缀 + target 的纠正 token。
        4. 全部一致：再额外补一个由 target model 给出的真实 next token。
        """
        assert len(draft_tokens) > 0, "draft_tokens 不能为空"

        target_tokens = self.target.verify_token_sequence(seq.token_ids, draft_tokens)

        accepted_prefix: list[int] = []
        for draft_token, target_token in zip(draft_tokens, target_tokens[:-1]):
            if draft_token == target_token:
                accepted_prefix.append(draft_token)
            else:
                return SpeculativeResult(
                    accepted_tokens=accepted_prefix + [target_token],
                    num_accepted_draft_tokens=len(accepted_prefix),
                    rejected=True,
                )

        tail_token = target_tokens[-1]
        return SpeculativeResult(
            accepted_tokens=accepted_prefix + [tail_token],
            num_accepted_draft_tokens=len(accepted_prefix),
            rejected=False,
        )
```

---

## 7. 修改 `engine/llm_engine.py`

### 7.1 speculative 主循环

新增两个方法到 `LLMEngine`：

```python
def _run_speculative_decode(self, seqs: list[Sequence]) -> list[list[int]]:
    """
    对一批 running 序列执行教学版 speculative decode。

    返回值与 seqs 一一对应，每条序列返回"本轮最终应当追加的 token 列表"。
    """
    assert self.draft_runner is not None
    assert self.speculative_verifier is not None

    outputs: list[list[int]] = []
    for seq in seqs:
        # 教学版验证是 greedy 匹配，只对 greedy 请求无损（§0 取舍一）
        assert seq.temperature == 0, (
            "教学版 speculative decoding 只支持 temperature=0 的请求；"
            "随机采样请求需要 §9 的接受采样版本"
        )
        draft_tokens = self.draft_runner.propose_tokens(seq, self.config.speculative_k)
        result = self.speculative_verifier.verify(seq, draft_tokens)
        outputs.append(result.accepted_tokens)
    return outputs
```

### 7.2 多 token 写回：块账本必须跟上

主线每轮 decode 只追加 1 个 token，块账本由 `schedule()` 里的 `can_append()/append_slot()` 推进。speculative 一轮可能追加 2~k+1 个 token——`schedule()` 只为**第 1 个**做过 `append_slot`，剩下的必须在写回时逐个补上，否则 `append_slot` 的块边界检测（`len(seq) % block_size == 1`）会被多 token 跳变绕过，块表和序列长度失配：

```python
def _postprocess_speculative_outputs(self, seqs: list[Sequence], speculative_outputs: list[list[int]]):
    """
    把 speculative decode 接受的多个 token 逐个写回 Sequence。

    第 i>0 个 token 追加前要先补块账本（schedule 只为本轮第 1 个新
    token 调过 append_slot）。块不够时提前截断本轮接受——正确性不受
    影响，只是这轮少接受几个。
    """
    finished_seqs = []
    for seq, accepted_tokens in zip(seqs, speculative_outputs):
        for i, token_id in enumerate(accepted_tokens):
            if i > 0:
                if not self.block_manager.can_append(seq):
                    break
                self.block_manager.append_slot(seq)
            done = self.scheduler.postprocess([seq], [token_id])
            if done:
                finished_seqs.extend(done)
                break
    return finished_seqs
```

`postprocess` 每 token 调一次，EOS / max_tokens 检查因此逐 token 生效——draft 猜出的 EOS 之后的 token 不会被误写回。

### 7.3 `step()` 的 decode 分支

在 Day8 版本的 `step()` 上改 decode 分支（prefill 分支保持 Day8 原样）：

```python
def step(self) -> tuple[list[tuple[int, list[int]]], int]:
    seqs, is_prefill, chunk_sizes = self.scheduler.schedule()
    if not seqs:
        return [], 0

    if is_prefill:
        # —— Day8 的 prefill 分支，原样保留 ——
        num_tokens = sum(chunk_sizes)
        token_ids = self.model_runner.run(seqs, True, chunk_sizes)
        self.scheduler.mark_prefill_progress(seqs, chunk_sizes)
        sampled_seqs, sampled_tokens = [], []
        for seq, token_id in zip(seqs, token_ids):
            if seq.prefill_done and seq.num_completion_tokens == 0:
                sampled_seqs.append(seq)
                sampled_tokens.append(token_id)
        finished_seqs = self.scheduler.postprocess(sampled_seqs, sampled_tokens)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
        return outputs, num_tokens

    # —— decode 分支：按配置选择 speculative 或普通路径 ——
    if self.config.enable_speculative_decoding:
        speculative_outputs = self._run_speculative_decode(seqs)
        finished_seqs = self._postprocess_speculative_outputs(seqs, speculative_outputs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
        return outputs, -len(seqs)

    token_ids = self.model_runner.run(seqs, False)
    finished_seqs = self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
    return outputs, -len(seqs)
```

一个诚实的说明：speculative 模式下，decode 从此**只走全量重算路径**（§0 取舍二），主线写进 KV cache 的内容不再被 decode 读取；prefill 照常写 cache 也无妨。块账本仍然要维护（§7.2），因为 `deallocate()`、`can_append()` 这些接口的正确性依赖它。

---

## 8. 装配：`LLMEngine.__init__()` 里创建 draft runner 和 verifier

前面几节只定义了零件，这一节把它们接上电。在 `LLMEngine.__init__()` 创建 `Scheduler` 之后追加：

```python
# engine/llm_engine.py —— __init__() 末尾追加
# 需要的 import（文件顶部）：
#   from engine.speculative import DraftModelRunner, SpeculativeVerifier

self.draft_runner = None
self.speculative_verifier = None
if self.config.enable_speculative_decoding:
    print(f"[LLMEngine] 加载草稿模型：{self.config.draft_model_path}")
    # 草稿模型用一个独立的 ModelRunner 加载。
    # 注意：不给它调用 allocate_kv_cache()——教学版 draft 前向
    # 走 forward_logits_no_cache()，全程不碰 KV cache。
    draft_config = Config(model_path=self.config.draft_model_path)
    self.draft_model_runner = ModelRunner(draft_config)

    self.draft_runner = DraftModelRunner(self.draft_model_runner)
    self.speculative_verifier = SpeculativeVerifier(self.model_runner)
```

要点：

1. `DraftModelRunner` 包的是**草稿模型**的 `ModelRunner`；`SpeculativeVerifier` 包的是**主模型**的 `self.model_runner`。两者都只用到 §5 新增的 no-cache 接口。
2. 草稿模型的 `ModelRunner` 不分配 KV cache（`kv_cache` 保持 `None`），也不需要 BlockManager——这正是 no-cache 设计换来的装配简单性。
3. 关闭开关时 `draft_runner / speculative_verifier` 是 `None`，`step()` 走普通 decode，主线行为零变化。

---

## 9. 延伸：从 greedy 匹配到 Leviathan 接受采样

想升级成论文版（对采样请求也无损），需要补两条数据通路：

1. `DraftModelRunner.propose_tokens` 除了 token，还要返回每步的**归一化概率分布** $q_i(\cdot)$（或至少 $q_i(x_i)$）。
2. `verify_token_sequence` 返回每个位置的 target **概率分布** $p_i(\cdot)$（softmax 后的 logits），而不是 argmax。

接受规则替换 `SpeculativeVerifier.verify` 里的逐位比较：

```python
# 对第 i 个 draft token x_i：
r = torch.rand(())
if r < min(1.0, p_i[x_i] / q_i[x_i]):
    accept(x_i)
else:
    # 从修正分布重采一个 token，然后停止
    residual = torch.clamp(p_i - q_i, min=0)
    residual = residual / residual.sum()
    corrected = torch.multinomial(residual, 1).item()
    accept(corrected); stop
# 全部接受后，额外从 p_{k+1} 采一个 tail token
```

数学结论（论文定理 1）：这样得到的每个 token 的边缘分布与直接从 target 采样**完全相同**。工程上多出来的成本主要是把两个 `[k, vocab]` 的概率张量从 draft/verify 前向里传出来。教学版先不做，是为了让 accept/reject 控制流保持一眼可读。

---

## 10. 新增 `tests/test_Day10_speculative.py`

不加载真实模型，专门锁 accept/reject 语义：

```python
"""Day10 speculative decoding 结构测试。"""

import sys

sys.path.insert(0, ".")

from engine.sequence import Sequence
from engine.speculative import SpeculativeVerifier, SpeculativeResult
from sampling_params import SamplingParams


class FakeTargetRunner:
    def __init__(self, verify_outputs):
        self.verify_outputs = verify_outputs

    def verify_token_sequence(self, prefix_token_ids, draft_tokens):
        return self.verify_outputs


def test_speculative_accept_all_then_append_tail():
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0))
    verifier = SpeculativeVerifier(FakeTargetRunner([10, 11, 12]))

    result = verifier.verify(seq, [10, 11])

    assert isinstance(result, SpeculativeResult)
    assert result.accepted_tokens == [10, 11, 12]
    assert result.num_accepted_draft_tokens == 2
    assert result.rejected is False


def test_speculative_reject_on_first_mismatch():
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0))
    verifier = SpeculativeVerifier(FakeTargetRunner([99, 88, 77]))

    result = verifier.verify(seq, [10, 11])

    assert result.accepted_tokens == [99]
    assert result.num_accepted_draft_tokens == 0
    assert result.rejected is True


def test_speculative_accept_prefix_then_reject():
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0))
    verifier = SpeculativeVerifier(FakeTargetRunner([10, 88, 77]))

    result = verifier.verify(seq, [10, 11])

    assert result.accepted_tokens == [10, 88]
    assert result.num_accepted_draft_tokens == 1
    assert result.rejected is True


def test_draft_runner_feeds_growing_context():
    from engine.speculative import DraftModelRunner

    class FakeDraftModelRunner:
        def __init__(self):
            self.calls = []

        def decode_one_token(self, token_ids):
            self.calls.append(list(token_ids))
            return 100 + len(token_ids)

    fake = FakeDraftModelRunner()
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0))
    draft = DraftModelRunner(fake)

    proposed = draft.propose_tokens(seq, 3)

    assert proposed == [103, 104, 105]
    # 每一步的上下文都应包含之前提议的 token
    assert fake.calls == [[1, 2, 3], [1, 2, 3, 103], [1, 2, 3, 103, 104]]


if __name__ == "__main__":
    test_speculative_accept_all_then_append_tail()
    test_speculative_reject_on_first_mismatch()
    test_speculative_accept_prefix_then_reject()
    test_draft_runner_feeds_growing_context()
    print("Day10 speculative tests passed")
```

---

## 11. 验收命令

```bash
python -m py_compile config.py engine/speculative.py engine/model_runner.py engine/llm_engine.py tests/test_Day10_speculative.py
python tests/test_Day10_speculative.py
```

有两份模型权重时的端到端手测（target 和 draft 可以先用同一个模型验证链路——此时 draft 全对，每轮应接受满 k+1 个 token）：

```bash
python - <<'PY'
from llm import LLM
from sampling_params import SamplingParams

llm = LLM(
    "models/Qwen3-0.6B",
    enable_speculative_decoding=True,
    draft_model_path="models/Qwen3-0.6B",
    speculative_k=4,
)
out = llm.generate(["1+1等于几？"], SamplingParams(temperature=0.0, max_tokens=32))
print(out[0]["text"])
PY
```

验收标准：输出与关闭 speculative 时的 greedy 输出**逐 token 一致**（greedy 匹配版的无损性就体现在这）。

---

## 12. 常见坑

1. **把 speculative decoding 塞进 `Sampler`。** `Sampler` 只管从 logits 选 token；accept/reject 是更高层的生成控制流。
2. **让 verifier 走随机采样，却不实现接受采样。** 那样 accept/reject 语义既不稳定也不无损。要么 greedy 匹配（本篇），要么完整的 Leviathan 接受采样（§9），没有中间态。
3. **多 token 写回不补块账本。** `append_slot` 的边界检测按"每轮 +1 token"设计，speculative 一轮 +k 会绕过它（§7.2）。
4. **给验证用的临时上下文分配 block。** 教学版验证前向不读写 cache，根本不需要 block——一旦分配就要操心释放，反而容易泄漏。
5. **以为这版能直接提速。** 不复用 KV cache 的教学版每轮全量重算，长上下文下比普通 decode 慢。它锁的是语义正确性，不是性能数字。
6. **绕过 EOS/max_tokens 检查。** 接受的 token 必须逐个过 `postprocess`，draft 猜到 EOS 后面的 token 不能写回。

---

## 13. 读完你应该明白

speculative decoding 是一条新的生成控制流，接入点是 `DraftModelRunner + SpeculativeVerifier + LLMEngine.step() 的 decode 分支`。教学版的验证是 greedy 精确匹配——对 greedy 请求无损、对采样请求不适用；升级到论文版要把 draft/target 的概率分布传出来做接受采样（§9）。教学版不复用 KV cache，换来的是零块账本负担和干净的控制流，代价是每轮全量重算。先把 accept/reject 语义写对，再谈性能。

下一篇：`Day11-MoE推理主线与专家并行认知篇.md`。

---

## 14. 文件级修改清单

| 文件 | 要写什么 | 别写什么 |
|---|---|---|
| `config.py` | 新增 `enable_speculative_decoding / draft_model_path / speculative_k` 显式开关和校验 | 别让 speculative decoding 默默改变默认单模型路径 |
| `engine/model_runner.py` | 新增 `forward_logits_no_cache() / decode_one_token() / verify_token_sequence()`：不读写 cache 的整段前向 + greedy 判定 | 别给临时上下文分配 block，别在 prefill 切片逻辑（run_model）上叠验证需求 |
| `engine/speculative.py` | 新增 `DraftModelRunner`、`SpeculativeVerifier`、`SpeculativeResult`，只管 draft/verify/accept-reject 控制流 | 别在这里改 Attention、KV Cache 布局或 Scheduler 状态机 |
| `engine/llm_engine.py` | `__init__` 装配 draft runner/verifier；decode 分支按配置分流；多 token 写回逐个补块账本、逐个过 postprocess | 别把 speculative 塞进 Scheduler，别绕过 EOS/max_tokens 检查 |
| `tests/test_Day10_speculative.py` | 不加载模型的语义测试：全接受、首 token 拒绝、前缀接受后拒绝、draft 上下文递增 | 别依赖真实草稿模型，别把性能提升当单元测试断言 |
