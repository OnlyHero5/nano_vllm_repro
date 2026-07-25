# Day 10 — Speculative Decoding：让小模型先猜，大模型来验

主线 Day0-7 解决的是”怎么把一个模型高效跑起来”。但它默认每轮 decode 只由主模型自己预测一个 token。

speculative decoding 打破这个默认：

1. 先让一个更便宜的 draft 模型一次猜 `k` 个 token。
2. 再让主模型一次验证这串 draft token。
3. 能接受的整段接受，不能接受的回退到主模型真实输出。

主模型很贵、草稿模型便宜时，这能显著减少主模型的单 token decode次数。对长生成来说，吞吐提升往往很明显。

关键认知：**speculative decoding 是一层新的生成控制流，不是新的 attention kernel。** 所以这次不碰 Attention、KV Cache 布局和 Scheduler 状态机——在 `Scheduler → ModelRunner → Sampler → postprocess` 这条链路外面包一层 draft-verify 逻辑就够了。Medusa、tree-based verifier、TP/CUDA Graph 联动、draft-target 共享 KV cache——都不做。

---

## 1. 一轮 speculative decode 长什么样

对每条 running 序列：

1. draft model 基于当前上下文，连续提议 `k` 个 token。
2. target model 一次性验证这 `k` 个位置。
3. 从左到右比较：draft token 与 target token 一致就接受；第一次不一致就停。
4. 全部接受？再补一个 target model 决定的真实 next token。
5. 中间拒绝？保留已接受的前缀，把第一个拒绝位置改成 target token。

---

## 2. 当前代码的三个相关点

### 2.1 `ModelRunner.run()` 只返回”每条序列一个 next token”

普通 decode 够用，但 speculative 的验证阶段需要一次处理”当前上下文 + 一段草稿 token”，得到整段位置上的 logits。

### 2.2 `Sampler` 职责很纯：从 logits 采样

别把 accept/reject 塞进 `Sampler`。它继续做单步采样，speculative 的比较逻辑放在更上层的 verifier。

### 2.3 `Sequence` 已经够用

不需要重写。教学版只需要在当前 step 内把”最终接受的 token 列表”逐个追加回 `Sequence`。

---

## 3. 两个模型，两个角色

- **target model**：当前仓库已有的主模型，结果以它为准。
- **draft model**：更小、更便宜，只负责提议 token。

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

当前仓库没有在线 serving config 系统、多模型 worker pool、API 级 feature flags，所以教学版只在 `Config` 里把主开关、草稿路径和 `k` 暴露清楚就够了。

---

## 5. 新增 `engine/speculative.py`

别把所有 speculative 逻辑塞进 `LLMEngine` 或 `ModelRunner`。新建一个文件，专门负责 draft 提议、target 验证、accept/reject 计算。下面这份可以直接写成 `engine/speculative.py`：

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

    教学版里它只暴露一个方法：
    propose_tokens(seq, k) -> list[int]
    """

    def __init__(self, llm_like_object):
        self.llm = llm_like_object

    @torch.inference_mode()
    def propose_tokens(self, seq: Sequence, k: int) -> list[int]:
        """
        基于当前 Sequence 提议 k 个草稿 token。

        教学版为了保持边界清楚，直接循环调用草稿模型的单步 decode 接口。
        这不是最终性能最优写法，但它最适合先把语义讲清楚。
        """
        assert k > 0, "k 必须 > 0"

        proposed: list[int] = []
        working_tokens = list(seq.token_ids)

        for _ in range(k):
            next_token = self.llm.decode_one_token(working_tokens)
            proposed.append(next_token)
            working_tokens.append(next_token)

        return proposed


class SpeculativeVerifier:
    """
    speculative decoding 的接受 / 拒绝逻辑。
    """

    def __init__(self, target_llm_like_object):
        self.target_llm = target_llm_like_object

    @torch.inference_mode()
    def verify(self, seq: Sequence, draft_tokens: list[int]) -> SpeculativeResult:
        """
        验证一段 draft token。

        规则：
        1. target model 一次性对 [context + draft_tokens] 做前向。
        2. 从左到右比较 target token 与 draft token。
        3. 如果第一次不一致，就停止继续接受。
        4. 如果全部一致，再额外补一个由 target model 给出的真实 next token。
        """
        assert len(draft_tokens) > 0, "draft_tokens 不能为空"

        target_tokens = self.target_llm.verify_token_sequence(seq.token_ids, draft_tokens)

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

## 6. 修改 `engine/model_runner.py`

当前 `ModelRunner` 已经有 `prepare_prefill()`、`prepare_decode()`、`run_model()`、`run()`。这次只新增一个”目标模型验证一段 token”的接口，加入 `ModelRunner` 类内部：

```python
@torch.inference_mode()
def decode_one_token(self, token_ids: list[int]) -> int:
    """
    给定一条完整上下文，返回下一个 token。

    教学版说明：
    - 这个接口主要给 DraftModelRunner 使用。
    - 它为了保持边界简单，直接临时构造一条 Sequence 并走普通 prefill 路径。
    - 这不是最终高性能版本，但它与当前仓库现有代码最兼容。
    """
    seq = Sequence(token_ids)
    self._ensure_prompt_blocks(seq)
    input_ids, positions = self.prepare_prefill([seq], [len(seq.prompt_token_ids)])

    try:
        logits = self.run_model(input_ids, positions, True)
        temperatures = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        next_token = self.sampler(logits, temperatures)[0].item()
        return int(next_token)
    finally:
        reset_context()


def _ensure_prompt_blocks(self, seq: Sequence) -> None:
    """
    确保一个临时 Sequence 拥有 block_table。

    普通主线里 block 分配由 Scheduler 管。
    speculative 的草稿模型临时推理不经过主调度器，所以这里需要一个最小 helper。
    """
    if not seq.block_table:
        self.block_manager.allocate(seq)


@torch.inference_mode()
def verify_token_sequence(self, prefix_token_ids: list[int], draft_tokens: list[int]) -> list[int]:
    """
    验证一段 draft token。

    输出长度为 len(draft_tokens) + 1：
    - 前 len(draft_tokens) 个位置对应 target 对 draft 位置的判断。
    - 最后 1 个位置对应 draft 全接受时的额外 next token。
    """
    full_tokens = list(prefix_token_ids) + list(draft_tokens)
    seq = Sequence(full_tokens)
    self._ensure_prompt_blocks(seq)

    input_ids, positions = self.prepare_prefill([seq], [len(seq.prompt_token_ids)])

    try:
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)

        start = len(prefix_token_ids) - 1
        end = len(prefix_token_ids) + len(draft_tokens)
        sliced_logits = logits[start:end]

        temperatures = torch.zeros(sliced_logits.shape[0], dtype=torch.float32, device=self.device)
        sampled = self.sampler(sliced_logits, temperatures)
        return sampled.tolist()
    finally:
        reset_context()
```

### 6.1 为什么 verify 用 greedy

target model 的验证阶段，本质上是在判断”它自己真正愿意走哪条 token 路径”。如果这里引入随机采样，accept/reject 语义就不稳定了。所以：**draft 可以是采样，verify 必须是确定性判断。**

---

## 7. 修改 `engine/llm_engine.py`

别把 speculative 逻辑塞进 `Scheduler`。Scheduler 继续只管"这一轮处理谁"，`LLMEngine.step()` 在 decode 分支里决定走普通 decode 还是 speculative decode。

新增两个方法到 `LLMEngine`，再改 `step()` 的 decode 分支：

```python
def _run_speculative_decode(self, seqs: list[Sequence]) -> list[list[int]]:
    """
    对一批 running 序列执行教学版 speculative decode。

    返回值：
    - 与 seqs 一一对应。
    - 每条序列返回“本轮最终应当追加的 token 列表”。
    """
    assert self.draft_runner is not None
    assert self.speculative_verifier is not None

    outputs: list[list[int]] = []
    for seq in seqs:
        draft_tokens = self.draft_runner.propose_tokens(seq, self.config.speculative_k)
        result = self.speculative_verifier.verify(seq, draft_tokens)
        outputs.append(result.accepted_tokens)
    return outputs


def _postprocess_speculative_outputs(self, seqs: list[Sequence], speculative_outputs: list[list[int]]):
    """
    把 speculative decode 接受的多个 token 逐个写回 Sequence。
    """
    finished_seqs = []
    for seq, accepted_tokens in zip(seqs, speculative_outputs):
        for token_id in accepted_tokens:
            done = self.scheduler.postprocess([seq], [token_id])
            if done:
                finished_seqs.extend(done)
                break
    return finished_seqs
```

`step()` 的 decode 分支改成：

```python
if self.config.enable_speculative_decoding:
    speculative_outputs = self._run_speculative_decode(seqs)
    finished_seqs = self._postprocess_speculative_outputs(seqs, speculative_outputs)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
    return outputs, -len(seqs)

# 普通 decode 路径保持原样。
token_ids = self.model_runner.run(seqs, False)
finished_seqs = self.scheduler.postprocess(seqs, token_ids)
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished_seqs]
return outputs, -len(seqs)
```

---

## 8. 新增 `tests/test_Day10_speculative.py`

不加载真实模型，专门锁 accept/reject 语义：

```python
"""Day10 speculative decoding 结构测试。"""

import sys

sys.path.insert(0, ".")

from engine.sequence import Sequence
from engine.speculative import SpeculativeVerifier, SpeculativeResult
from sampling_params import SamplingParams


class FakeTargetLLM:
    def __init__(self, verify_outputs):
        self.verify_outputs = verify_outputs

    def verify_token_sequence(self, prefix_token_ids, draft_tokens):
        return self.verify_outputs


def test_speculative_accept_all_then_append_tail():
    seq = Sequence([1, 2, 3], SamplingParams())
    verifier = SpeculativeVerifier(FakeTargetLLM([10, 11, 12]))

    result = verifier.verify(seq, [10, 11])

    assert isinstance(result, SpeculativeResult)
    assert result.accepted_tokens == [10, 11, 12]
    assert result.num_accepted_draft_tokens == 2
    assert result.rejected is False


def test_speculative_reject_on_first_mismatch():
    seq = Sequence([1, 2, 3], SamplingParams())
    verifier = SpeculativeVerifier(FakeTargetLLM([99, 88, 77]))

    result = verifier.verify(seq, [10, 11])

    assert result.accepted_tokens == [99]
    assert result.num_accepted_draft_tokens == 0
    assert result.rejected is True


def test_speculative_accept_prefix_then_reject():
    seq = Sequence([1, 2, 3], SamplingParams())
    verifier = SpeculativeVerifier(FakeTargetLLM([10, 88, 77]))

    result = verifier.verify(seq, [10, 11])

    assert result.accepted_tokens == [10, 88]
    assert result.num_accepted_draft_tokens == 1
    assert result.rejected is True
```

---

## 9. 验收命令

```bash
python -m py_compile config.py engine/speculative.py engine/model_runner.py engine/llm_engine.py tests/test_Day10_speculative.py
python tests/test_Day10_speculative.py
```

快速看 accept/reject 语义：

```bash
python - <<'PY'
from engine.sequence import Sequence
from engine.speculative import SpeculativeVerifier
from sampling_params import SamplingParams

class FakeTarget:
    def verify_token_sequence(self, prefix_token_ids, draft_tokens):
        return [10, 88, 77]

seq = Sequence([1, 2, 3], SamplingParams())
verifier = SpeculativeVerifier(FakeTarget())
result = verifier.verify(seq, [10, 11])
print(result)
PY
```

---

## 10. 常见坑

1. **把 speculative decoding 塞进 `Sampler`。** `Sampler` 只管从 logits 选 token；accept/reject 是更高层的生成控制流。
2. **让 verifier 走随机采样。** accept/reject 语义会不稳定。
3. **让 draft 和 target 立即共享 KV cache。** 边界复杂度一下抬太高。
4. **在 `Scheduler` 里重写 speculative 逻辑。** Scheduler 继续管”这一轮处理谁”，speculative 的额外控制流放 `LLMEngine` 外层。
5. **以为重点是”多采样几个 token”。** 真正的重点是草稿提议、target 验证、正确的接受/拒绝回退语义。

---

## 11. 读完你应该明白

speculative decoding 是一条新的生成控制流，不是新的 attention kernel。教学版最稳的接入点是 `DraftModelRunner + SpeculativeVerifier + LLMEngine.decode 分支`。verifier 的职责是判断 draft token 是否和 target model 的真实偏好一致。先把 accept/reject 语义写对，再想性能优化。

下一篇：`Day11-MoE推理主线与专家并行认知篇.md`。

---

## 12. 文件级修改清单

| 文件 | 要写什么 | 别写什么 |
|---|---|---|
| `config.py` | 新增 `enable_speculative_decoding / draft_model_path / speculative_k` 显式开关和校验 | 别让 speculative decoding 默默改变默认单模型路径 |
| `engine/speculative.py` | 新增 `DraftModelRunner`、`SpeculativeVerifier`、`SpeculativeResult`，只管 draft/verify/accept-reject 控制流 | 别在这里改 Attention、KV Cache 布局或 Scheduler 状态机 |
| `engine/model_runner.py` | 补草稿单步提议和 target 批量验证的最小接口，复用现有前向/logits/sampler 边界 | 别让 verifier 走随机采样，别让 draft/target 立即共享 KV cache |
| `engine/llm_engine.py` | decode 分支按配置选择普通 decode 或 speculative decode，接受的多个 token 逐个交给 postprocess | 别把 speculative 塞进 Scheduler，别绕过 EOS/max_tokens 检查 |
| `tests/test_Day10_speculative.py` | 不加载模型的语义测试：全接受、首 token 拒绝、前缀接受后拒绝 | 别依赖真实草稿模型，别把性能提升当单元测试断言 |
