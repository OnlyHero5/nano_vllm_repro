# 07. 补齐 Benchmark 和 Day7 验收

这一篇是收尾。

目标是：

> 用真实脚本测一次当前实现，并把 README / TODO 改成和真实代码一致。

本篇要补四件事：

1. 新增 `bench.py`。
2. 新增轻量测试 `tests/test_Day7.py`。
3. 回写 `readme.md`。
4. 回写 `todo_list.md`。

注意顺序：

```text
先写 benchmark
  ↓
再跑 benchmark
  ↓
拿真实结果回写 README
  ↓
最后整理 TODO
```

不要先写性能结论，再回头找数据。

---

## 1. 前置条件

默认前面已经完成：

1. 单卡 `LLM.generate()` 能跑。
2. `generate()` 返回 `list[dict]`。
3. `SamplingParams` 支持 `temperature / top_k / top_p / max_tokens`。

如果单卡链路还没跑通，不要急着 benchmark。

---

## 2. 当前仓库还缺什么

当前缺：

- `bench.py`
- `tests/test_Day7.py`
- README 中真实可用的命令
- README 中真实 benchmark 表格
- 和当前代码状态一致的 `todo_list.md`

测试和 benchmark 要分开：

- `tests/test_Day7.py` 只测结构，必须轻量。
- `bench.py` 真的跑模型，可以比较重。

---

## 3. 新增 `bench.py`

下面是一个基础版 benchmark 结构。

它做的事很简单：

1. 构造同一批 prompts。
2. 用 nano 后端或 HF 后端跑生成。
3. 记录总耗时和吞吐。
4. 输出 Markdown 表格或 JSON。

```python
"""nano-vllm / Hugging Face 基础 benchmark。

这不是论文级 benchmark。
它的目标是给当前仓库一个可重复的本地测量入口。
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from llm import LLM
from sampling_params import SamplingParams


@dataclass
class BenchmarkResult:
    """一次 benchmark 的聚合结果。"""

    backend: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int

    # 基础版暂时不单独 hook 首 token 事件。
    # 没有真实 TTFT 数据时，不要编造；先写 0.0。
    ttft_ms: float

    total_latency_ms: float
    throughput_tps: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano-vllm / HF benchmark")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--backend", choices=["nano", "hf", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    return parser


def build_prompts(batch_size: int) -> list[str]:
    """构造一批固定 prompt，保证多次运行条件一致。"""
    seed_prompts = [
        "请用一句话解释 PagedAttention。",
        "请用一句话解释 Continuous Batching。",
        "请用一句话解释 Prefix Cache。",
        "请用一句话解释 CUDA Graph 为什么常用于 decode。",
    ]

    prompts: list[str] = []
    while len(prompts) < batch_size:
        prompts.extend(seed_prompts)
    return prompts[:batch_size]


def count_prompt_tokens(tokenizer, prompts: list[str]) -> int:
    """统计 prompt token 数，用同一个 tokenizer 口径。"""
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    return int(encoded["attention_mask"].sum().item())


def make_sampling_params(args) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def aggregate_result(
    backend: str,
    batch_size: int,
    prompt_tokens: int,
    output_tokens: int,
    total_seconds: list[float],
) -> BenchmarkResult:
    """把多次 repeat 的耗时合成一行结果。"""
    avg_total = mean(total_seconds)
    throughput = 0.0 if avg_total == 0 else output_tokens / avg_total

    return BenchmarkResult(
        backend=backend,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=0.0,
        total_latency_ms=avg_total * 1000.0,
        throughput_tps=throughput,
    )


def format_markdown_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Backend | Batch | Prompt Tokens | Output Tokens | TTFT (ms) | Total (ms) | Throughput (tok/s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item.backend} | {item.batch_size} | {item.prompt_tokens} | "
            f"{item.output_tokens} | {item.ttft_ms:.2f} | "
            f"{item.total_latency_ms:.2f} | {item.throughput_tps:.2f} |"
        )
    return "\n".join(lines)
```

### 3.1 nano 后端

```python
def run_nano_backend(args, prompts: list[str], tokenizer) -> BenchmarkResult:
    """
    跑当前仓库的 LLM.generate。

    注意：
    这里每次 repeat 都复用同一个 llm 实例，
    避免把模型加载时间算进生成耗时。
    """
    llm = LLM(args.model_path)
    sampling_params = make_sampling_params(args)

    for _ in range(args.warmup):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    total_seconds: list[float] = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        total_seconds.append(time.perf_counter() - start)

        # generate 返回 list[dict]，每个 dict 里有 token_ids。
        output_tokens = sum(len(item["token_ids"]) for item in outputs)

    prompt_tokens = count_prompt_tokens(tokenizer, prompts)
    return aggregate_result(
        backend="nano",
        batch_size=args.batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_seconds=total_seconds,
    )
```

### 3.2 HF 后端

```python
@torch.inference_mode()
def run_hf_backend(args, prompts: list[str], tokenizer) -> BenchmarkResult:
    """
    跑 Hugging Face generate。

    nano 和 HF 要尽量使用同一组输入和采样参数，
    否则 benchmark 结果没有比较意义。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    generate_kwargs = dict(
        max_new_tokens=args.max_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
    )
    if args.top_k > 0:
        generate_kwargs["top_k"] = args.top_k

    for _ in range(args.warmup):
        model.generate(**inputs, **generate_kwargs)

    total_seconds: list[float] = []
    output_tokens = 0

    for _ in range(args.repeat):
        start = time.perf_counter()
        outputs = model.generate(**inputs, **generate_kwargs)
        total_seconds.append(time.perf_counter() - start)

        output_tokens = int(outputs.shape[1] - inputs["input_ids"].shape[1]) * args.batch_size

    prompt_tokens = int(inputs["attention_mask"].sum().item())
    return aggregate_result(
        backend="hf",
        batch_size=args.batch_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_seconds=total_seconds,
    )
```

### 3.3 `main()`

```python
def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = build_prompts(args.batch_size)

    results: list[BenchmarkResult] = []

    if args.backend in {"nano", "both"}:
        results.append(run_nano_backend(args, prompts, tokenizer))

    if args.backend in {"hf", "both"}:
        results.append(run_hf_backend(args, prompts, tokenizer))

    if args.json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        print(format_markdown_table(results))


if __name__ == "__main__":
    main()
```

---

## 4. 新增 `tests/test_Day7.py`

这个测试只测 benchmark 的结构，不加载模型。

```python
"""Day7 benchmark 结构测试。"""

import sys
sys.path.insert(0, ".")

from bench import (
    BenchmarkResult,
    aggregate_result,
    build_parser,
    format_markdown_table,
)


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.model_path == "models/Qwen3-0.6B"
    assert args.backend == "both"
    assert args.batch_size == 4
    assert args.max_tokens == 64


def test_aggregate_result_metrics():
    result = aggregate_result(
        backend="nano",
        batch_size=4,
        prompt_tokens=120,
        output_tokens=40,
        total_seconds=[1.0, 1.2],
    )

    assert isinstance(result, BenchmarkResult)
    assert result.backend == "nano"
    assert result.batch_size == 4
    assert result.prompt_tokens == 120
    assert result.output_tokens == 40
    assert result.ttft_ms == 0.0
    assert result.total_latency_ms == 1100.0
    assert round(result.throughput_tps, 2) == 36.36


def test_markdown_table_format():
    result = BenchmarkResult(
        backend="hf",
        batch_size=2,
        prompt_tokens=64,
        output_tokens=32,
        ttft_ms=0.0,
        total_latency_ms=400.0,
        throughput_tps=80.0,
    )

    table = format_markdown_table([result])

    assert "| Backend | Batch | Prompt Tokens | Output Tokens |" in table
    assert "| hf | 2 | 64 | 32 | 0.00 | 400.00 | 80.00 |" in table
```

---

## 5. 回写 `readme.md`

README 只写事实。

至少改四处：

1. 目录树改成当前仓库真实布局，不要继续写成上游 `nanovllm/` 包布局。
2. 快速运行命令改成：

```bash
python example.py
```

3. benchmark 命令写成：

```bash
python bench.py --backend both --batch_size 4 --max_tokens 64 --repeat 3
```

4. 性能表只放真实输出。

性能表上方加一句：

```markdown
> 以下数据来自本机实际测试，只代表当前硬件、模型、依赖版本和参数配置。
```

如果还没跑 benchmark，就不要写“提升多少倍”。

可以先写：

```markdown
| Backend | Batch | Prompt Tokens | Output Tokens | TTFT (ms) | Total (ms) | Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| 待实测 | - | - | - | - | - | - |
```

---

## 6. 回写 `todo_list.md`

建议改成三段：

```markdown
# TODO

## 已完成

- [x] Day1 基础数据结构
- [x] Day2 Qwen3 模型骨架
- [x] Day3 BlockManager / Attention 主线
- [x] Day4 Scheduler / ModelRunner 基础链路

## 正在补齐

- [ ] Day5 单卡 generate 主循环验收
- [ ] Day6 Tensor Parallel 基础版
- [ ] Day6 CUDA Graph 基础版
- [ ] Day7 benchmark 和 README 回写

## 下一步

- 跑完整测试。
- 跑 benchmark。
- 根据真实结果更新 README。
```

勾选状态要按真实代码来，不要按计划来。

---

## 7. 验收命令

先测结构：

```bash
python -m py_compile bench.py tests/test_Day7.py
python tests/test_Day7.py
```

再跑真实 benchmark：

```bash
python bench.py --backend both --batch_size 4 --max_tokens 64 --repeat 3
```

如果机器显存不够，先缩小参数：

```bash
python bench.py --backend nano --batch_size 1 --max_tokens 16 --repeat 1
```

---

## 8. 常见坑

1. **先改 README，后跑 benchmark**
   很容易写出没有数据支撑的结论。

2. **在 `tests/test_Day7.py` 里跑大模型**
   测试会太重，不适合常跑。

3. **nano 和 HF 用不同 prompt 或采样参数**
   结果不能比较。

4. **TTFT 没真实测量却写成真实数值**
   基础版没有 hook 首 token 事件时，就明确写 0 或“不测”。

5. **TODO 勾选和代码状态不一致**
   这会让后续读者不知道项目到底做到哪一步。

---

## 9. 本篇结束后你应该明白

Day7 的重点不是“有一个 benchmark 文件”。

真正的重点是：

1. 测试和 benchmark 分层。
2. README 写真实命令和真实结果。
3. TODO 反映代码现状。
4. 性能结论必须来自实际运行数据。
