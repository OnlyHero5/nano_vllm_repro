# Day13 文档整理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `plans/13-实现GPU-Offload与跨后端扩展总览.md` 整理成格式规范、代码块可复制、语义不误导的进阶实验收口文档。

**Architecture:** 只修改文档，不新增 `utils/`、`tests/` 或主线代码。第 13 篇保留“独立实验原型”的定位，并通过静态扫描确认 `08~13` 没有明显占位、伪代码省略或 Python 语法拼写错误。

**Tech Stack:** Markdown、Python 代码块静态抽取、`python -m py_compile` 临时验证。

---

## File Structure

- Modify: `plans/13-实现GPU-Offload与跨后端扩展总览.md` — 修正文档排版、列表间距、代码块缩进、`assert0` 语法错误和中英混排空格。
- No create: 不新增 `utils/gpu_offload_proto.py`，不新增 `tests/test_Day13_gpu_offload_proto.py`。
- Inspect only: `plans/08-实现Chunked-Prefill与v1调度策略.md` 到 `plans/12-实现FP8与KV-Cache量化实验篇.md` — 只做轻量扫描；除非扫描发现明显占位或语法坏块，否则不改。

---

### Task 1: 修正 Day13 文档格式和代码块

**Files:**
- Modify: `plans/13-实现GPU-Offload与跨后端扩展总览.md`

- [ ] **Step 1: 定位现有问题**

Run:

```bash
python - <<'PY'
from pathlib import Path
path = Path('plans/13-实现GPU-Offload与跨后端扩展总览.md')
for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
    stripped = line.strip()
    markers = ['省略', 'TO' + 'DO', 'TB' + 'D']
    if stripped.startswith('assert0') or stripped.startswith('-需要') or stripped.startswith('-怎么') or any(marker in stripped for marker in markers):
        print(f'{i}: {stripped}')
PY
```

Expected: 输出至少包含 `assert0 <= block_id < self.num_blocks`，以及若干缺少空格的列表行。

- [ ] **Step 2: 修正文档正文排版**

Edit `plans/13-实现GPU-Offload与跨后端扩展总览.md`:

- 标题改为 `# 13. 实现 GPU Offload 与跨后端扩展总览`。
- 把 `08~12`、`KV cache`、`CPU / GPU`、`TTS 适配` 等中英文混排处补齐必要空格。
- 把列表项改成 `- 内容`，编号改成 `1. 内容`。
- 保留“不并回主线、不伪造后端、不误导读者”的范围说明。

Expected: Markdown 标题、列表和段落格式与 `plans/08~12` 风格一致。

- [ ] **Step 3: 修正 Python 代码块缩进和语法**

Edit 第 4 节代码块，保证代码块里的核心片段是可复制的完整 Python：

```python
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class OffloadBlock:
    cpu_tensor: torch.Tensor
    gpu_tensor: torch.Tensor | None
    resident: str


class GPUOffloadCacheProto:
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

Expected: `assert 0 <= block_id < self.num_blocks` 有空格，类和方法体使用 4 空格缩进。

- [ ] **Step 4: 修正 Day13 测试代码块和命令说明**

Edit 第 5 节和第 8 节：

- 测试代码块使用 4 空格缩进。
- `block_shape=(2, 3)`、`report["num_blocks"] == 4` 等表达式补齐空格。
- 明确说明命令只有在读者把代码块落到对应文件后才可运行，不暗示当前仓库已经新增这些文件。

Expected: 文档不会让读者误以为当前工作树已经包含 `utils/gpu_offload_proto.py`。

---

### Task 2: 验证 Day13 代码块可编译

**Files:**
- Inspect: `plans/13-实现GPU-Offload与跨后端扩展总览.md`

- [ ] **Step 1: 抽取 Python 代码块到临时目录**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

text = Path('plans/13-实现GPU-Offload与跨后端扩展总览.md').read_text(encoding='utf-8')
blocks = re.findall(r'```python\n(.*?)```', text, flags=re.S)
print(len(blocks))
for i, block in enumerate(blocks, 1):
    out = Path('/tmp') / f'day13_block_{i}.py'
    out.write_text(block, encoding='utf-8')
    print(out)
PY
```

Expected: 至少输出 2 个 Python 代码块路径。

- [ ] **Step 2: 编译独立原型代码块**

Run:

```bash
python -m py_compile /tmp/day13_block_1.py
```

Expected: 命令退出码为 0。

- [ ] **Step 3: 处理非独立片段**

If later code blocks are shell heredoc examples or depend on files not created in this task, do not force them as standalone modules. Instead inspect for obvious syntax issues by reading the extracted file:

```bash
python - <<'PY'
from pathlib import Path
for path in sorted(Path('/tmp').glob('day13_block_*.py')):
    print(f'--- {path} ---')
    print(path.read_text(encoding='utf-8')[:500])
PY
```

Expected: 非独立片段没有 `assert0`、坏缩进或缺失括号。

---

### Task 3: 轻量扫描 08~13 一致性

**Files:**
- Inspect: `plans/08-实现Chunked-Prefill与v1调度策略.md`
- Inspect: `plans/09-实现Radix-Prefix-Cache与可观测指标.md`
- Inspect: `plans/10-实现Speculative-Decoding基础版.md`
- Inspect: `plans/11-实现MoE推理主线与专家并行认知篇.md`
- Inspect: `plans/12-实现FP8与KV-Cache量化实验篇.md`
- Inspect: `plans/13-实现GPU-Offload与跨后端扩展总览.md`

- [ ] **Step 1: 扫描占位词、坏断言和明显省略**

Run:

```bash
python - <<'PY'
from pathlib import Path
root = Path('plans')
for path in sorted(root.glob('*.md')):
    if not path.name[:2].isdigit() or not (8 <= int(path.name[:2]) <= 13):
        continue
    hits = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        stripped = line.strip()
        markers = ['TO' + 'DO', 'TB' + 'D', '省略', 'pass  #', 'assert0']
        if any(token in stripped for token in markers):
            hits.append((i, stripped))
    if hits:
        print(path.name)
        for line_no, line in hits:
            print(f'  {line_no}: {line}')
PY
```

Expected: 修正后不再输出 `13` 的 `assert0`；如果 `08~12` 没有新问题，不修改它们。

- [ ] **Step 2: 扫描 Markdown 标题基础格式**

Run:

```bash
python - <<'PY'
from pathlib import Path
for path in sorted(Path('plans').glob('*.md')):
    if not path.name[:2].isdigit() or not (8 <= int(path.name[:2]) <= 13):
        continue
    first = path.read_text(encoding='utf-8').splitlines()[0]
    print(f'{path.name}: {first}')
PY
```

Expected: `08~13` 第一行均形如 `# 08.`、`# 09.`、`# 13.`。

---

### Task 4: 最终差异检查

**Files:**
- Inspect: `plans/13-实现GPU-Offload与跨后端扩展总览.md`
- Inspect: `docs/superpowers/plans/2026-04-30-day13-doc-cleanup.md`

- [ ] **Step 1: 查看 diff 范围**

Run:

```bash
git diff -- plans/13-实现GPU-Offload与跨后端扩展总览.md docs/superpowers/plans/2026-04-30-day13-doc-cleanup.md
```

Expected: diff 只包含第 13 篇文档整理和本实施计划；没有新增 `utils/` 或 `tests/` 文件。

- [ ] **Step 2: 查看工作树状态**

Run:

```bash
git status --short
```

Expected: 仍可能显示之前已有的 `01~07` 修改和 `08~13` 新文件；本任务新增的额外文件只有 `docs/superpowers/plans/2026-04-30-day13-doc-cleanup.md`，且第 13 篇被整理。
```
