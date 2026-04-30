# 03. 补全 SamplingParams 和 Sampler

这一篇补采样能力。

目标是：

> 让每条请求都能带自己的 `temperature / top_k / top_p`，并让 `Sampler` 同时兼容旧调用和新调用。

改完后要支持：

- `temperature=0` 表示 greedy。
- `top_k=0` 表示不启用 top-k。
- `top_p=1.0` 表示不启用 top-p。
- 每条 `Sequence` 都保存自己的采样参数。

---

## 1. 当前代码是什么状态

当前主要问题：

1. `SamplingParams` 只有 `temperature / max_tokens / ignore_eos`。
2. `SamplingParams` 还把 `temperature=0` 当非法值。
3. `Sequence` 没保存 `top_k / top_p`。
4. `Sampler.forward()` 只接收 `temperatures`。
5. Day1、Day4 测试还停在旧接口。

本篇要做的是“兼容升级”，不是一口气把所有调用点都改爆。

---

## 2. 修改 `sampling_params.py`

把 `SamplingParams` 改成下面这个含义：

```python
@dataclass
class SamplingParams:
    """
    一条请求的采样配置。

    这些字段会先进入 Sequence，
    然后由 ModelRunner 在每个 step 里整理成张量传给 Sampler。
    """

    # temperature=0 表示 greedy，不做随机采样。
    temperature: float = 1.0

    # top_k=0 表示不启用 top-k。
    # top_k>0 表示只保留概率最高的 K 个 token。
    top_k: int = 0

    # top_p=1.0 表示不启用 top-p。
    # top_p<1.0 表示只保留累计概率达到 p 的最小 token 集合。
    top_p: float = 1.0

    # 最多生成多少个新 token。
    max_tokens: int = 4096

    # True 表示即使遇到 eos 也继续生成，直到 max_tokens。
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        assert self.temperature >= 0.0, "temperature 必须 >= 0"
        assert self.top_k >= 0, "top_k 必须 >= 0"
        assert 0.0 < self.top_p <= 1.0, "top_p 必须在 (0, 1] 内"
        assert self.max_tokens > 0, "max_tokens 必须 > 0"
```

注意这里的三个约定：

1. `temperature=0` 合法。
2. `top_k=0` 合法。
3. `top_p=1.0` 合法。

---

## 3. 修改 `engine/sequence.py`

`Sequence` 是请求进入系统后的运行期状态。

所以采样参数不能只留在用户传入的 `SamplingParams` 里，必须复制到 `Sequence`。

在 `Sequence.__init__()` 里，把采样字段改成：

```python
# 这些字段来自 SamplingParams。
# 后面的 Scheduler / ModelRunner 都只看 Sequence，
# 不会再回头找用户原始入参。
self.temperature = sampling_params.temperature
self.top_k = sampling_params.top_k
self.top_p = sampling_params.top_p
self.max_tokens = sampling_params.max_tokens
self.ignore_eos = sampling_params.ignore_eos
```

---

## 4. 修改 `layers/sampler.py`

### 4.1 保持旧调用可用

`Sampler.forward()` 建议写成：

```python
def forward(
    self,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor | None = None,
    top_ps: torch.Tensor | None = None,
) -> torch.Tensor:
```

这样旧代码仍然可以调用：

```python
sampler(logits, temperatures)
```

新代码也可以调用：

```python
sampler(logits, temperatures, top_ks, top_ps)
```

进入函数后先补默认值：

```python
if top_ks is None:
    # 旧调用没有传 top_k，就等价于不启用 top-k。
    top_ks = torch.zeros_like(temperatures, dtype=torch.long)

if top_ps is None:
    # 旧调用没有传 top_p，就等价于不启用 top-p。
    top_ps = torch.ones_like(temperatures, dtype=torch.float32)
```

### 4.2 增加 top-k 裁剪

```python
def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    只保留分数最高的 top_k 个 token。

    top_k <= 0 表示不启用。
    top_k >= vocab_size 也等价于不启用。
    """
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    values, _ = torch.topk(logits, k=top_k, dim=-1)

    # 第 top_k 大的值就是保留门槛。
    threshold = values[..., -1, None]

    # 小于门槛的 token 填成 -inf。
    # softmax 后它们的概率就是 0。
    return logits.masked_fill(logits < threshold, float("-inf"))
```

### 4.3 增加 top-p 裁剪

```python
def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus sampling。

    先按概率从大到小排序，
    再保留累计概率刚好覆盖 top_p 的最小集合。
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # True 表示这个位置要被丢掉。
    sorted_mask = cumulative_probs > top_p

    # 右移一位，是为了保留“刚刚让累计概率超过 top_p 的那个 token”。
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()

    # 概率最高的 token 永远保留，避免极小 top_p 把所有 token 都删掉。
    sorted_mask[..., 0] = False

    masked_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # 排序空间里的结果要散回原始 vocab 索引空间。
    restored = torch.full_like(masked_sorted_logits, float("-inf"))
    restored.scatter_(dim=-1, index=sorted_indices, src=masked_sorted_logits)
    return restored
```

### 4.4 `forward()` 的推荐流程

```python
def forward(
    self,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor | None = None,
    top_ps: torch.Tensor | None = None,
) -> torch.Tensor:
    if top_ks is None:
        top_ks = torch.zeros_like(temperatures, dtype=torch.long)
    if top_ps is None:
        top_ps = torch.ones_like(temperatures, dtype=torch.float32)

    # temperature=0 表示 greedy。
    greedy_mask = temperatures == 0

    # 避免除 0。greedy 的位置后面会直接 argmax，所以这里临时设成 1。
    safe_temperatures = temperatures.clone()
    safe_temperatures[greedy_mask] = 1.0

    # temperature 是缩放 logits，不是缩放 softmax 后的概率。
    scaled_logits = logits.float() / safe_temperatures.unsqueeze(dim=1)

    # 每条样本可以有自己的 top_k/top_p，所以逐行处理。
    filtered_rows = []
    for row_logits, top_k, top_p in zip(
        scaled_logits,
        top_ks.tolist(),
        top_ps.tolist(),
    ):
        row_logits = self._apply_top_k(row_logits, int(top_k))
        row_logits = self._apply_top_p(row_logits, float(top_p))
        filtered_rows.append(row_logits)

    filtered_logits = torch.stack(filtered_rows, dim=0)
    probs = torch.softmax(filtered_logits, dim=-1)

    # Gumbel-Max 写法。它等价于按 probs 随机采样一个 token。
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    sampled_tokens = (probs / noise).argmax(dim=-1)

    # greedy 样本直接取最大 logits。
    greedy_tokens = filtered_logits.argmax(dim=-1)
    return torch.where(greedy_mask, greedy_tokens, sampled_tokens)
```

---

## 5. 修改测试

### 5.1 Day1：`temperature=0` 不再非法

旧测试如果写了“`temperature=0` 应该报错”，要改掉。

推荐测试：

```python
sp = SamplingParams()
assert sp.temperature == 1.0
assert sp.top_k == 0
assert sp.top_p == 1.0

sp = SamplingParams(temperature=0.0, top_k=20, top_p=0.9, max_tokens=128)
assert sp.temperature == 0.0
assert sp.top_k == 20
assert sp.top_p == 0.9
assert sp.max_tokens == 128
```

非法值测试应该改成：

```python
for kwargs in [
    {"temperature": -1.0},
    {"top_k": -1},
    {"top_p": 0.0},
    {"top_p": 1.1},
    {"max_tokens": 0},
]:
    try:
        SamplingParams(**kwargs)
        raise AssertionError(f"应该拒绝非法参数: {kwargs}")
    except AssertionError:
        pass
```

### 5.2 Day4：同时测旧接口和新接口

```python
sampler = Sampler()
logits = torch.randn(4, 1000)
temps = torch.tensor([0.0, 0.5, 1.0, 2.0])

# 旧接口仍然能用。
tokens = sampler(logits, temps)
assert tokens.shape == (4,)
assert tokens[0] == logits[0].argmax()

# 新接口也能用。
top_ks = torch.tensor([0, 10, 20, 50])
top_ps = torch.tensor([1.0, 0.9, 0.8, 0.95])
tokens = sampler(logits, temps, top_ks, top_ps)
assert tokens.shape == (4,)
```

---

## 6. 验收命令

```bash
python -m py_compile sampling_params.py engine/sequence.py layers/sampler.py
python tests/test_Day1.py
python tests/test_Day4.py
```

快速手测：

```bash
python - <<'PY'
import torch
from layers.sampler import Sampler

sampler = Sampler()
logits = torch.randn(2, 100)
temps = torch.tensor([0.0, 0.8])
top_ks = torch.tensor([0, 10])
top_ps = torch.tensor([1.0, 0.9])

print(sampler(logits, temps, top_ks, top_ps))
PY
```

---

## 7. 常见坑

1. **继续把 `temperature=0` 当非法值**
   这会和 greedy 语义冲突。

2. **把 `top_k / top_p` 写成全局单值**
   batch 内不同请求就不能用不同采样策略。

3. **top-p 后忘记恢复原始 vocab 顺序**
   采样会在排序后的索引空间里进行，结果会错。

4. **一次性强制所有调用点都改四参数**
   不利于定位问题。先兼容，再迁移。

---

## 8. 本篇结束后你应该明白

这一篇的重点是：

1. 采样参数属于每条请求，不是全局开关。
2. `temperature / top_k / top_p` 分别控制不同层面的随机性。
3. 接口升级最好先兼容旧调用，再逐步接新调用。

下一篇进入单卡推理主循环：

- `04-补齐单卡推理链路与Day5测试.md`
