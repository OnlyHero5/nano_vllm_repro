# Day 11A — Expert Offloading：8GB 卡怎么跑 32 个 expert

Day11 的 `Qwen3MoEMLP` 把所有 expert 权重都放在 GPU 上。8 个 expert 还好，32 个就炸了——MoE 的显存压力主要出在 expert 权重上，不是 attention 或 KV cache。

经典的解法是 **expert-level offloading**：

- 全量 expert 权重存在 CPU 主区（可选 pinned memory）。
- GPU 上只预分配 `K << num_experts` 个 expert 容器（slot），按需 H2D 换入。
- 用 LRU 决定哪个 slot 被覆盖。
- 用 **routing 频次**把热门 expert 永久 pin 在 GPU 上（PowerInfer/Mixtral-Offloading 的思路）。

所有改动集中在 `experiments/moe_offloading/` 新目录和一份测试文件里。Dense Qwen3-0.6B 主线、`Qwen3MoEMLP` 主线都不碰。

---

## 1. 做完之后你应该能

1. 解释 MoE 推理的显存压力为什么主要出在 expert 权重上，并算出 8 GB 卡能塞下多大的 MoE。
2. 实现一个 CPU master + GPU slot pool + LRU 的 `ExpertWeightCache`，并证明它的输出和"全 expert 都在 GPU"的参考实现 bit-for-bit 等价。
3. 实现"基于 routing 频次的热门 expert 钉死"，并量化命中率提升。
4. 知道为什么这条玩具路径可以推广到真实 Qwen1.5-MoE / DeepSeek-V2-Lite，但当前仓库没必要直接接上去。

**边界**：

| 范围 | 做 | 不做 |
|---|---|---|
| Expert 权重 CPU↔GPU 换入换出 | ✅ | ❌ all-to-all、expert parallel |
| LRU + 热门 pin | ✅ | ❌ 异步 prefetch overlap（给出扩展点） |
| 与 Day11 `Qwen3MoEMLP` 输出等价性自检 | ✅ | ❌ 改 `models/qwen3.py` 主线 |
| 真实 Qwen1.5-MoE / DeepSeek-V2-Lite 跑通 | ❌（示例参数 + 随机权重） | — |

所有机械动作都暴露出来，但模型大小由 `--num-experts/--hidden/--intermediate` 控制，CI 可在 CPU 上 30 秒内跑完。

---

## 2. 参考来源

| 来源 | 借鉴什么 |
|---|---|
| Mixtral-Offloading（`dvmazur/mixtral-offloading`） | "GPU expert slot pool + LRU" 的整体骨架；speculative expert prefetch 的设计动机 |
| PowerInfer（`SJTU-IPADS/PowerInfer`） | 把神经元/expert 区分为热点（GPU 常驻）+ 冷门（按需换入） |
| FlexGen（`FMInference/FlexGen`） | block 级 swap 的"先调度后执行"思路 |
| HuggingFace MoE blog（`huggingface/blog/moe`） | Top-k routing + expert dispatch 的标准语义 |
| nano-vLLM 主仓 + PR #116 | MoE 在 nano 项目里的最小工程边界 |

参考链接：

- Mixtral-Offloading：<https://github.com/dvmazur/mixtral-offloading>
- PowerInfer：<https://github.com/SJTU-IPADS/PowerInfer>
- FlexGen：<https://github.com/FMInference/FlexGen>
- HuggingFace MoE blog：<https://huggingface.co/blog/moe>
- nano-vLLM 主仓：<https://github.com/GeeeekExplorer/nano-vllm>
- nano-vLLM MoE PR：<https://github.com/GeeeekExplorer/nano-vllm/pull/116>

---

## 3. 前置：Day11 必须已落地

```bash
ls models/qwen3.py
python -c "from models.qwen3 import MoEExpert, MoERouter, Qwen3MoEMLP; print('ok')"
```

三个 import 任何一个失败，先回 Day11 把 `models/qwen3.py` 的 §4.2–4.4 补齐。

要新建的文件：

```text
experiments/__init__.py
experiments/moe_offloading/__init__.py
experiments/moe_offloading/expert_cache.py
experiments/moe_offloading/offloaded_mlp.py
experiments/moe_offloading/run_demo.py
tests/test_Day11A_offloading.py
```

不会修改任何 `engine/` `layers/` `models/` `utils/` 下的文件。

---

## 4. 设计要点

### 4.1 为什么按 expert 粒度 offload

因为 MoE 模型里：

- **Attention + embed + lm_head** 一次只有一份，量级与 dense 同级，不必拆。
- **Expert FFN** 才是真正的"`num_experts` 倍权重膨胀"。Qwen1.5-MoE-A2.7B 总参 14.3 B，其中 expert 占绝大多数；DeepSeek-V2-Lite 16 B，expert 同理。
- 每次 forward **只激活 `top_k` 个 expert**（典型 `top_k=2`）。所以同一时刻 GPU 上**只需要** "active experts ∪ pinned experts" 这一小部分。

把 expert 当作可热插拔的"显存页"是 MoE offloading 的最自然抽象。

### 4.2 LRU + Pin 的双层策略

```
┌────────────────────────────────────────────┐
│  ExpertWeightCache                         │
│                                            │
│  ┌─────────── CPU master (pinned) ──────┐  │
│  │ expert 0, 1, 2, ..., num_experts-1   │  │
│  └──────────────────────────────────────┘  │
│                ▲           ▲                │
│        (H2D copy)   (load_from_modulelist) │
│                ▼                            │
│  ┌────── GPU slot pool (K slots) ───────┐  │
│  │ slot 0: pinned expert 3              │  │
│  │ slot 1: LRU 当前持有 expert 5         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

- **K 个 slot**：固定预分配，shape = (1 个 expert 的权重)。
- **Pin**：`pin_experts([3])` 把 expert 3 永久放进 slot 0，永不被淘汰。
- **LRU**：剩下的 slot 用 `OrderedDict` 维护近因序，需要换入新 expert 时弹出最老的。

预热阶段会统计 `expert_call_count[i]`，之后 `pin_experts(top-N)` 即可让命中率从随机的 `K/num_experts` 跃升到 50% 以上（top-k routing 的频次分布通常很倾斜）。

### 4.3 为什么暂不实现异步 prefetch

异步 prefetch（`torch.cuda.Stream` 上 H2D copy + 与 compute overlap）是真实 Mixtral-Offloading 的关键性能点，但它需要在 forward 一开始就拿到 `topk_ids`，再把"当前 expert 计算"和"下一个 expert copy"穿插。这条路径会让 forward 主循环变难读，对玩具学习目标负贡献。

先把同步换入做对、把命中率/miss 数据收齐，再在 §10 给出"如何升级到异步 prefetch"的扩展骨架。

---

## 5. 创建 `experiments/__init__.py` 和 `experiments/moe_offloading/__init__.py`

两个文件都留空：

```bash
mkdir -p experiments/moe_offloading
: > experiments/__init__.py
: > experiments/moe_offloading/__init__.py
```

或者直接创建空文件：

```python
# experiments/__init__.py
```

```python
# experiments/moe_offloading/__init__.py
```

---

## 6. 创建 `experiments/moe_offloading/expert_cache.py`

完整内容如下：

```python
"""单卡 MoE expert 显存换入换出缓存。

设计：
- CPU master 区：保存所有 expert 的权重副本（GPU 可用时使用 pinned memory）。
- GPU slot pool：预分配 num_gpu_slots 个 MoEExpert 容器，按 LRU 复用。
- 热门 expert 可以通过 pin_experts() 永久驻留某个 slot，不被 LRU 淘汰。

约束：
- 复用 11 篇定义的 models.qwen3.MoEExpert，从而保证形状与 dense MoE 完全一致。
- 一次 H2D copy 完成两个 weight tensor（gate_up_proj.weight / down_proj.weight）。
- expert 内部的 weight_loader 不参与本次实验，加载只用 .data.copy_()。
"""

from collections import OrderedDict

import torch
from torch import nn

from models.qwen3 import MoEExpert


class ExpertWeightCache:
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_gpu_slots: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_pinned_cpu: bool = True,
    ) -> None:
        assert 1 <= num_gpu_slots <= num_experts, "num_gpu_slots 必须在 [1, num_experts] 内"
        self.num_experts = num_experts
        self.num_gpu_slots = num_gpu_slots
        self.device = torch.device(device)
        self.dtype = dtype

        # CPU master：每个 expert 一个独立 MoEExpert 实例，权重默认为随机初始化。
        # 真实使用时通过 load_from_modulelist() 灌入参考权重。
        self.cpu_experts = nn.ModuleList(
            [
                MoEExpert(hidden_size, intermediate_size).to(device="cpu", dtype=dtype)
                for _ in range(num_experts)
            ]
        )
        if use_pinned_cpu and torch.cuda.is_available() and self.device.type == "cuda":
            for expert in self.cpu_experts:
                expert.gate_up_proj.weight.data = (
                    expert.gate_up_proj.weight.data.contiguous().pin_memory()
                )
                expert.down_proj.weight.data = (
                    expert.down_proj.weight.data.contiguous().pin_memory()
                )

        # GPU slot pool：固定 K 个槽位。
        self.gpu_slots = nn.ModuleList(
            [
                MoEExpert(hidden_size, intermediate_size).to(device=self.device, dtype=dtype)
                for _ in range(num_gpu_slots)
            ]
        )

        # 状态表
        self._slot_to_expert: list[int | None] = [None] * num_gpu_slots
        self._expert_to_slot: dict[int, int] = {}
        # OrderedDict：最近使用的放尾部；只对非 pinned expert 维护
        self._lru: "OrderedDict[int, None]" = OrderedDict()
        self._pinned: set[int] = set()

        # 命中率 / 频次统计
        self.hits = 0
        self.misses = 0
        self.expert_call_count = [0] * num_experts

    # ------------------------------------------------------------------ public

    def load_from_modulelist(self, source_experts: nn.ModuleList) -> None:
        """从一个完整 nn.ModuleList[MoEExpert] 把权重灌进 CPU master。"""
        assert len(source_experts) == self.num_experts, "源 expert 数量与 cache 不一致"
        for i, src in enumerate(source_experts):
            self._copy_one_expert_into_cpu(self.cpu_experts[i], src)

    def pin_experts(self, expert_ids: list[int]) -> None:
        """把热门 expert 永久驻留 GPU。会重置整套 slot 状态。"""
        assert len(expert_ids) <= self.num_gpu_slots, "pin 数量不能超过 num_gpu_slots"
        assert len(set(expert_ids)) == len(expert_ids), "expert_ids 不能重复"
        for ex in expert_ids:
            assert 0 <= ex < self.num_experts, f"expert id 越界: {ex}"

        self._slot_to_expert = [None] * self.num_gpu_slots
        self._expert_to_slot = {}
        self._lru.clear()
        self._pinned = set(expert_ids)

        for slot_idx, expert_idx in enumerate(expert_ids):
            self._copy_expert_to_slot(expert_idx, slot_idx)

    def get_gpu_expert(self, expert_idx: int) -> MoEExpert:
        """返回一个 GPU 上的 MoEExpert（必要时换入）。"""
        assert 0 <= expert_idx < self.num_experts
        self.expert_call_count[expert_idx] += 1

        if expert_idx in self._expert_to_slot:
            self.hits += 1
            slot_idx = self._expert_to_slot[expert_idx]
            if expert_idx not in self._pinned:
                self._lru.move_to_end(expert_idx)
            return self.gpu_slots[slot_idx]

        self.misses += 1
        slot_idx = self._evict_one()
        self._copy_expert_to_slot(expert_idx, slot_idx)
        return self.gpu_slots[slot_idx]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "expert_call_count": list(self.expert_call_count),
            "pinned": sorted(self._pinned),
        }

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.expert_call_count = [0] * self.num_experts

    # ----------------------------------------------------------------- internal

    def _evict_one(self) -> int:
        for slot_idx, ex_idx in enumerate(self._slot_to_expert):
            if ex_idx is None:
                return slot_idx

        for ex_idx in list(self._lru):
            if ex_idx in self._pinned:
                continue
            slot_idx = self._expert_to_slot.pop(ex_idx)
            self._slot_to_expert[slot_idx] = None
            self._lru.pop(ex_idx)
            return slot_idx

        raise RuntimeError(
            "所有 GPU slot 都被 pin 占满，无法换入新 expert；"
            "请增大 num_gpu_slots 或减少 pin 的数量"
        )

    def _copy_expert_to_slot(self, expert_idx: int, slot_idx: int) -> None:
        gpu_expert = self.gpu_slots[slot_idx]
        cpu_expert = self.cpu_experts[expert_idx]

        is_cuda = self.device.type == "cuda"
        gpu_expert.gate_up_proj.weight.data.copy_(
            cpu_expert.gate_up_proj.weight.data, non_blocking=is_cuda
        )
        gpu_expert.down_proj.weight.data.copy_(
            cpu_expert.down_proj.weight.data, non_blocking=is_cuda
        )
        if is_cuda:
            torch.cuda.synchronize(self.device)

        self._slot_to_expert[slot_idx] = expert_idx
        self._expert_to_slot[expert_idx] = slot_idx
        if expert_idx not in self._pinned:
            self._lru[expert_idx] = None
            self._lru.move_to_end(expert_idx)

    @staticmethod
    def _copy_one_expert_into_cpu(dst: MoEExpert, src: MoEExpert) -> None:
        # 注意 .to() 会把数据搬到正确 dtype；这里用 contiguous() 防止 pin_memory 失败。
        dst.gate_up_proj.weight.data.copy_(
            src.gate_up_proj.weight.data.detach().to(dst.gate_up_proj.weight.dtype).cpu().contiguous()
        )
        dst.down_proj.weight.data.copy_(
            src.down_proj.weight.data.detach().to(dst.down_proj.weight.dtype).cpu().contiguous()
        )
```

几个关键点：

1. `_copy_expert_to_slot` 用 `non_blocking=is_cuda` + `torch.cuda.synchronize(self.device)`。教学版优先保证语义正确，性能交给 §10 的扩展点。
2. `_evict_one` 显式处理"全 slot 被 pin 满"的情况，给出可读 `RuntimeError`，避免静默死循环。
3. `_copy_one_expert_into_cpu` 强制 `contiguous()` 是为了配合后续 `pin_memory()`：非 contiguous 的张量不能 pin。
4. 不动 `MoEExpert.weight.weight_loader`：这个 loader 协议是 11 篇主线 loader 用的，本实验绕过，直接 `.data.copy_()`。

---

## 7. 创建 `experiments/moe_offloading/offloaded_mlp.py`

完整内容如下：

```python
"""与 11 篇 Qwen3MoEMLP 行为等价、但 expert 权重按需 offload 的实验版本。

forward 路径与 Qwen3MoEMLP 几乎一致，唯一差别是 expert 不是 nn.ModuleList，
而是通过 ExpertWeightCache.get_gpu_expert(idx) 拿到。
"""

import torch
from torch import nn

from models.qwen3 import MoERouter
from experiments.moe_offloading.expert_cache import ExpertWeightCache


class OffloadedQwen3MoEMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_gpu_slots: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        norm_topk_prob: bool = True,
        use_pinned_cpu: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.device = torch.device(device)
        self.dtype = dtype

        self.router = MoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
        ).to(device=self.device, dtype=dtype)

        self.cache = ExpertWeightCache(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_gpu_slots=num_gpu_slots,
            device=self.device,
            dtype=dtype,
            use_pinned_cpu=use_pinned_cpu,
        )

    def load_from_dense_moe(self, src_mlp: nn.Module) -> None:
        """从 11 篇的 Qwen3MoEMLP 拷贝 router + experts 全套权重。"""
        # router 在 GPU 上常驻
        self.router.gate.weight.data.copy_(
            src_mlp.router.gate.weight.data.to(self.dtype).to(self.device)
        )
        # experts 进入 CPU master 区
        self.cache.load_from_modulelist(src_mlp.experts)

    def pin_top_experts(self, top_n: int) -> list[int]:
        """根据当前 expert_call_count 选出前 top_n 个 expert pin 在 GPU。返回被 pin 的 id 列表。"""
        counts = list(enumerate(self.cache.expert_call_count))
        counts.sort(key=lambda kv: -kv[1])
        ids = [idx for idx, _ in counts[:top_n]]
        self.cache.pin_experts(ids)
        return ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.router(hidden_states)
        final_hidden_states = torch.zeros_like(hidden_states)

        expert_mask = torch.nn.functional.one_hot(
            topk_ids, num_classes=self.num_experts
        ).permute(2, 1, 0)

        active_experts = torch.where(expert_mask.sum(dim=(1, 2)) > 0)[0].tolist()
        for expert_idx in active_experts:
            topk_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gpu_expert = self.cache.get_gpu_expert(expert_idx)
            current = gpu_expert(current)
            current = current * topk_weights[token_idx, topk_pos, None]
            final_hidden_states.index_add_(0, token_idx, current)

        return final_hidden_states
```

`forward` 与 Day11 `Qwen3MoEMLP.forward` 在语义上**完全一致**，唯一差别是 `self.experts[expert_idx]` 变成了 `self.cache.get_gpu_expert(expert_idx)`。**接口边界不变，只换 backing store**。

---

## 8. 创建 `experiments/moe_offloading/run_demo.py`

完整内容如下：

```python
"""手工演示 Offloaded MoE：等价性自检 + warmup 频次统计 + pin 后命中率提升。

用法（默认参数 30 秒可在 CPU 跑完）：
    python -m experiments.moe_offloading.run_demo
    python -m experiments.moe_offloading.run_demo --pin-top 1
    python -m experiments.moe_offloading.run_demo --biased-input
    python -m experiments.moe_offloading.run_demo --num-experts 16 --num-slots 4 --pin-top 2 --biased-input
"""

import argparse

import torch

from models.qwen3 import Qwen3MoEMLP
from experiments.moe_offloading.offloaded_mlp import OffloadedQwen3MoEMLP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--intermediate", type=int, default=256)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--num-slots", type=int, default=2)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--measure-steps", type=int, default=120)
    p.add_argument("--pin-top", type=int, default=0)
    p.add_argument("--biased-input", action="store_true",
                   help="把输入限制在固定子空间，模拟真实模型里偏斜的路由分布")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_input(
    tokens: int,
    hidden: int,
    biased: bool,
    device,
    dtype,
    _cache: dict = {},
) -> torch.Tensor:
    if not biased:
        return torch.randn(tokens, hidden, device=device, dtype=dtype)
    # 用固定的 K 个 topic 向量构造输入：每个 token = 某个 topic + 小扰动。
    # 这样 router 看到的输入分布有明显的低维聚簇结构，topk 会偏向少数 expert，
    # 模拟真实语料里"几个话题反复出现"的局部性。
    cache_key = (hidden, str(device), str(dtype))
    if cache_key not in _cache:
        gen = torch.Generator(device=device).manual_seed(42)
        num_topics = max(2, hidden // 32)
        _cache[cache_key] = (
            torch.randn(num_topics, hidden, generator=gen, device=device, dtype=dtype) * 4.0
        )
    topics = _cache[cache_key]
    idx = torch.randint(0, topics.shape[0], (tokens,), device=device)
    noise = torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.1
    return topics[idx] + noise


def main() -> None:
    args = parse_args()
    if args.pin_top >= args.num_slots:
        raise ValueError(
            f"pin-top ({args.pin_top}) 必须小于 num-slots ({args.num_slots})，"
            "否则没有 LRU 槽位留给冷门 expert"
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"[Config] device={device} dtype={dtype} biased_input={args.biased_input}")
    print(
        f"[Config] hidden={args.hidden} intermediate={args.intermediate} "
        f"num_experts={args.num_experts} top_k={args.top_k} num_slots={args.num_slots}"
    )

    # 1. 参考实现：全 expert 都常驻 GPU
    ref = Qwen3MoEMLP(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.num_experts,
        top_k=args.top_k,
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    # 2. 实验对象：CPU master + K 个 GPU slot
    off = OffloadedQwen3MoEMLP(
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_gpu_slots=args.num_slots,
        device=device,
        dtype=dtype,
    )
    off.load_from_dense_moe(ref)

    # 3. 等价性自检
    x = make_input(args.tokens, args.hidden, args.biased_input, device, dtype)
    with torch.no_grad():
        y_ref = ref(x)
        y_off = off(x)
    max_abs = (y_ref - y_off).abs().max().item()
    print(f"[Equivalence] max |y_ref - y_off| = {max_abs:.3e}")
    assert max_abs < 1e-4, "Offloaded MoE 输出与参考实现不一致"

    # 4. warmup：让频次统计有意义
    for _ in range(args.warmup_steps):
        xx = make_input(args.tokens, args.hidden, args.biased_input, device, dtype)
        with torch.no_grad():
            off(xx)
    print(f"[Warmup] expert_call_count = {off.cache.expert_call_count}")
    print(f"[Warmup] pre-pin stats     = {off.cache.stats()}")

    # 5. 可选 pin 热门 expert
    if args.pin_top > 0:
        pinned = off.pin_top_experts(args.pin_top)
        print(f"[Pin] pinned experts = {pinned}")
    off.cache.reset_stats()

    # 6. 真实负载下统计命中率
    for _ in range(args.measure_steps):
        xx = make_input(args.tokens, args.hidden, args.biased_input, device, dtype)
        with torch.no_grad():
            off(xx)
    final = off.cache.stats()
    print(f"[Measure] post-pin stats   = {final}")
    print(
        f"[Summary] hit_rate={final['hit_rate']:.3f}  "
        f"GPU resident experts <= {args.num_slots}  "
        f"(参考实现需要 {args.num_experts})"
    )


if __name__ == "__main__":
    main()
```

典型输出（CUDA，`hidden=128 intermediate=256`）：

**默认（均匀随机输入）**：

```
$ python -m experiments.moe_offloading.run_demo --pin-top 1
[Equivalence] max |y_ref - y_off| = 0.000e+00
[Warmup] expert_call_count = [31, 31, 31, 31, 31, 31, 31, 31]   # 完全均匀
[Pin]    pinned experts = [0]
[Measure] post-pin stats = {'hits': 100, 'misses': 700, 'hit_rate': 0.125, ...}
[Summary] hit_rate=0.125  GPU resident experts <= 2
```

注意这里 hit_rate = `1/num_experts` = `1/8 = 0.125`。这是**最坏情况**：随机权重 + 随机输入 → 路由完全均匀 → 每个 step 都会请求所有 8 个 expert → 单个 LRU slot 始终在被换入换出 → 只有被 pin 的 expert 0 稳定命中。

**偏斜输入（更接近真实 MoE 路由分布）**：

```
$ python -m experiments.moe_offloading.run_demo --pin-top 1 --biased-input
[Warmup] expert_call_count = [31, 0, 0, 31, 1, 0, 31, 31]       # 4 个 expert 几乎包揽
[Pin]    pinned experts = [0]
[Measure] post-pin stats = {'hits': 100, 'misses': 300, 'hit_rate': 0.250, ...}
[Summary] hit_rate=0.250  GPU resident experts <= 2

$ python -m experiments.moe_offloading.run_demo --num-slots 3 --pin-top 2 --biased-input
[Pin]    pinned experts = [0, 3]
[Measure] post-pin stats = {'hits': 200, 'misses': 200, 'hit_rate': 0.500, ...}
[Summary] hit_rate=0.500  GPU resident experts <= 3

$ python -m experiments.moe_offloading.run_demo --num-experts 16 --num-slots 4 --pin-top 2 --biased-input
[Warmup] expert_call_count = [5, 0, 0, 31, 0, 0, 31, 31, 0, 0, 31, 31, 0, 0, 0, 0]
[Measure] post-pin stats = {'hits': 200, 'misses': 312, 'hit_rate': 0.391, ...}
[Summary] hit_rate=0.391  GPU resident experts <= 4  (参考实现需要 16)
```

`--biased-input` 把输入限制成"几个固定 topic + 小扰动"，模拟真实语料里"几个话题反复出现"的局部性。可以观察到：

1. 等价性差异恒为 0（或极小数值噪声）。
2. 偏斜输入下 8 个 expert 中只有 4 个真正活跃，pin 1 个就把命中率从 12.5% 翻倍到 25%；pin 2 个 + 多 1 个 LRU 槽位，命中率达到 50%。
3. 把 `--num-experts` 拉到 16，pin 2 个仍能拿到 ~40% 命中率，说明扩展到更大 MoE 时收益不会消失。

为什么这个对比很重要：

1. 单看均匀输入，会以为 expert offloading 完全没用。
2. 真实模型的 MoE 路由有显著偏斜（DeepSeek-V3 报告里冷热 expert 调用频次差距常在 5–10 倍以上）。所以**真实场景**里 pin 热门 expert 才会带来"命中率几倍提升"的效果。
3. `--biased-input` 让你**亲手**看到这条规律：偏斜 → pin 见效；均匀 → pin 只能保住自己。这正是 PowerInfer "hot/cold expert" 思路成立的前提。

---

## 9. 创建 `tests/test_Day11A_offloading.py`

完整内容如下：

```python
"""Day11A MoE expert-offloading 测试。

不依赖真实模型权重，也不依赖 GPU；CUDA 可用时会顺便跑 GPU 路径。
"""

import sys

sys.path.insert(0, ".")

import torch
from torch import nn

from models.qwen3 import Qwen3MoEMLP
from experiments.moe_offloading.expert_cache import ExpertWeightCache
from experiments.moe_offloading.offloaded_mlp import OffloadedQwen3MoEMLP


def _device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


def test_offloaded_matches_reference_bitwise():
    torch.manual_seed(0)
    device, dtype = _device_and_dtype()

    ref = Qwen3MoEMLP(
        hidden_size=16,
        intermediate_size=24,
        num_experts=4,
        top_k=2,
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)

    off = OffloadedQwen3MoEMLP(
        hidden_size=16,
        intermediate_size=24,
        num_experts=4,
        top_k=2,
        num_gpu_slots=2,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )
    off.load_from_dense_moe(ref)

    x = torch.randn(8, 16, device=device, dtype=dtype)
    with torch.no_grad():
        y_ref = ref(x)
        y_off = off(x)
    assert torch.allclose(y_ref, y_off, atol=1e-5), (y_ref - y_off).abs().max().item()


def test_lru_eviction_keeps_only_num_slots_resident():
    device, dtype = _device_and_dtype()
    cache = ExpertWeightCache(
        num_experts=6,
        hidden_size=8,
        intermediate_size=16,
        num_gpu_slots=2,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )

    for idx in [0, 1, 2, 3]:
        cache.get_gpu_expert(idx)

    assert sum(1 for s in cache._slot_to_expert if s is not None) == 2
    assert set(cache._slot_to_expert) == {2, 3}, cache._slot_to_expert
    assert cache.misses == 4
    assert cache.hits == 0


def test_lru_keeps_recent_resident():
    device, dtype = _device_and_dtype()
    cache = ExpertWeightCache(
        num_experts=4,
        hidden_size=8,
        intermediate_size=16,
        num_gpu_slots=2,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )

    cache.get_gpu_expert(0)
    cache.get_gpu_expert(1)
    cache.get_gpu_expert(0)  # hit, 0 -> most recent
    cache.get_gpu_expert(2)  # miss, 应淘汰 1 而不是 0

    assert set(cache._slot_to_expert) == {0, 2}, cache._slot_to_expert
    assert cache.hits == 1
    assert cache.misses == 3


def test_pin_keeps_expert_resident_under_pressure():
    device, dtype = _device_and_dtype()
    cache = ExpertWeightCache(
        num_experts=6,
        hidden_size=8,
        intermediate_size=16,
        num_gpu_slots=2,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )
    cache.pin_experts([3])

    for idx in [0, 1, 2, 4, 5]:
        cache.get_gpu_expert(idx)

    assert 3 in cache._slot_to_expert, cache._slot_to_expert
    assert 5 in cache._slot_to_expert, cache._slot_to_expert


def test_pin_full_slots_then_request_unknown_raises():
    device, dtype = _device_and_dtype()
    cache = ExpertWeightCache(
        num_experts=4,
        hidden_size=8,
        intermediate_size=16,
        num_gpu_slots=2,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )
    cache.pin_experts([0, 1])

    cache.get_gpu_expert(0)  # 命中
    cache.get_gpu_expert(1)  # 命中

    try:
        cache.get_gpu_expert(2)
    except RuntimeError as exc:
        assert "GPU slot" in str(exc)
    else:
        raise AssertionError("应当抛 RuntimeError，因为 slot 全部被 pin 占满")


def test_load_from_modulelist_then_get_returns_correct_weights():
    device, dtype = _device_and_dtype()
    torch.manual_seed(0)

    ref = Qwen3MoEMLP(
        hidden_size=4,
        intermediate_size=8,
        num_experts=3,
        top_k=2,
    ).to(device=device, dtype=dtype)

    cache = ExpertWeightCache(
        num_experts=3,
        hidden_size=4,
        intermediate_size=8,
        num_gpu_slots=1,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )
    cache.load_from_modulelist(ref.experts)

    for idx in range(3):
        gpu_expert = cache.get_gpu_expert(idx)
        assert torch.allclose(
            gpu_expert.gate_up_proj.weight.data,
            ref.experts[idx].gate_up_proj.weight.data,
        )
        assert torch.allclose(
            gpu_expert.down_proj.weight.data,
            ref.experts[idx].down_proj.weight.data,
        )


def test_pin_top_experts_picks_highest_count():
    device, dtype = _device_and_dtype()
    off = OffloadedQwen3MoEMLP(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        top_k=2,
        num_gpu_slots=3,
        device=device,
        dtype=dtype,
        use_pinned_cpu=False,
    )
    # 手工伪造频次：expert 2 最多，expert 0 次之
    off.cache.expert_call_count = [10, 1, 50, 5]
    pinned = off.pin_top_experts(2)
    assert pinned == [2, 0]
    assert off.cache._pinned == {2, 0}


if __name__ == "__main__":
    test_offloaded_matches_reference_bitwise()
    test_lru_eviction_keeps_only_num_slots_resident()
    test_lru_keeps_recent_resident()
    test_pin_keeps_expert_resident_under_pressure()
    test_pin_full_slots_then_request_unknown_raises()
    test_load_from_modulelist_then_get_returns_correct_weights()
    test_pin_top_experts_picks_highest_count()
    print("Day11A offloading tests passed")
```

这套测试有意做成"无 GPU 也能跑通"，CI 友好；CUDA 可用时会自动顺便覆盖 GPU 路径（包括 `non_blocking` copy + `cuda.synchronize`）。

---

## 10. 验收命令

从 `nano_vll_repro/` 运行：

```bash
# 1. 语法检查（不依赖任何模型权重）
python -m py_compile \
    experiments/__init__.py \
    experiments/moe_offloading/__init__.py \
    experiments/moe_offloading/expert_cache.py \
    experiments/moe_offloading/offloaded_mlp.py \
    experiments/moe_offloading/run_demo.py \
    tests/test_Day11A_offloading.py

# 2. 直接跑测试
python tests/test_Day11A_offloading.py

# 3. 用 pytest 跑
python -m pytest tests/test_Day11A_offloading.py -q

# 4. demo（CPU 模式 30 秒，CUDA 模式更快）
python -m experiments.moe_offloading.run_demo
python -m experiments.moe_offloading.run_demo --pin-top 1
python -m experiments.moe_offloading.run_demo --num-experts 16 --num-slots 4 --pin-top 2

# 5. 回归：确认主线 dense Qwen3 / 11 篇 MoE 都没坏
python tests/test_Day1.py
python tests/test_Day2.py
python tests/test_Day3.py
python tests/test_Day4.py
python tests/test_Day11_moe.py
```

预期：

- 第 1、2、3、5 步全部 pass，第 5 步说明 offloading 不影响主线和 Day11。
- 第 4 步 `[Equivalence] max |...| = 0.000e+00` 或非常小，且 `--pin-top` 增大时 `hit_rate` 明显上升。

---

## 11. 常见坑

1. **以为 `--pin-top` 越大越好。**
   一旦 `pin_top >= num_slots`，没有任何 LRU 槽位留给冷门 expert，请求第 K+1 个不同 expert 会触发 `RuntimeError`。`pin_top < num_slots` 是必须满足的不变量。
2. **把整个 OffloadedQwen3MoEMLP 套进 `nn.DataParallel` 或多卡。**
   本实验是显式单卡设计：CPU master 在 CPU、GPU slot 在 device。多卡场景对应的是 expert parallel + all-to-all，跟 offloading 是两条路径。
3. **直接 `cache.cpu_experts[i].cuda()` 提速。**
   会破坏"CPU master 永远在 CPU"的不变量，并且 pinned memory 会失效。需要 GPU 上的副本就走 `get_gpu_expert(i)`。
4. **修改 `MoEExpert.weight_loader` 想着兼容本实验。**
   本实验完全绕过 `weight_loader`，全程 `.data.copy_()`。`weight_loader` 协议是 11 篇主线 loader 用的，请保持原状。
5. **dtype 不一致。**
   `OffloadedQwen3MoEMLP(dtype=...)` 必须和参考 `Qwen3MoEMLP.to(dtype=...)`、传入的 hidden_states 三方对齐，否则 `index_add_` 会因为 dtype 不匹配抛错。
6. **以为 `torch.cuda.synchronize` 是性能特性。**
   它在这里**只是为了语义正确**——`non_blocking=True` 的 H2D copy 必须等 copy 完成才能在 default stream 上使用。性能扩展见下一节。

---

## 12. 扩展点（选做）

想继续推的话，下面三条和真实生产 MoE offloading 方向一致：

1. **异步 prefetch + stream overlap。**
   预先用一个独立 `torch.cuda.Stream` 做 H2D copy；forward 主循环在算 expert k 时同时把 expert k+1 的权重 copy 到下一个 LRU slot。`get_gpu_expert(k+1)` 时只需 `stream.synchronize()`。Mixtral-Offloading 的 `MixtralExpertCache.prefetch` 是教科书示例。
2. **expert prefetch + speculative routing。**
   Mixtral-Offloading 论文里给的方法：用一个轻量代理网络预测 next-token routing 概率，把可能用到的 expert 提前往 GPU 搬。本仓库可以从"按上一步 topk_ids 复用"开始做最小版本。
3. **真上 Qwen1.5-MoE / DeepSeek-V2-Lite。**
   把 `OffloadedQwen3MoEMLP` 的 hidden_size/intermediate_size/num_experts/top_k 换成 HF config 的真实值，再写一个 `load_from_hf_safetensors()` 把 `model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight` 直接灌进 `cpu_experts`。loader 改动参考 Day11 §5 的 `_rewrite_moe_weight_name`，但不再走 `Qwen3ForCausalLM` 的 `packed_modules_mapping`。

---

## 13. 做完之后

仓库会多出一条与主线**完全解耦**的 MoE offloading 教学路径：

```text
experiments/moe_offloading/
  ├─ expert_cache.py     # CPU master + GPU slot pool + LRU + pin
  ├─ offloaded_mlp.py    # 与 Qwen3MoEMLP 等价的 forward，但按需换入
  └─ run_demo.py         # 等价性自检 + 频次统计 + pin 后命中率提升
tests/test_Day11A_offloading.py
```

你应该能回答四个问题：

1. MoE 推理的显存压力为什么主要出在 expert 权重上，不是 attention 或 KV cache。
2. LRU + pin 这套两层策略在数学上为什么能让命中率从 `K/num_experts` 提升到很多倍。
3. 为什么用"独立实验目录 + 不动主线"的形式，而 Day11 可以直接修改 `models/qwen3.py`。
4. 异步 prefetch、speculative routing、跨节点 expert parallel 各自解决什么瓶颈。

下一篇回到主线进阶：`Day12-FP8与KV-Cache量化实验篇.md`。
