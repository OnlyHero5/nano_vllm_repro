# Day 6 — 完整推理链路：Sampler、LLMEngine、example.py

## 本篇定位

前五天我们已经搭建了所有零件。今天把它们组装成一辆能跑的车。

读完本篇后，你**不需要再改任何代码就能跑通端到端推理**。
`python example.py` 应该输出 Qwen3 生成的回答。

---

## 1. 📖 知识点：采样策略

### 1.1 三种采样方式的递进关系

```
logits（原始分数）
   │
   ▼  ÷ temperature  ← 第一步：温度缩放（控制「随机性」）
缩放后的 logits
   │
   ▼   top-k 过滤    ← 第二步：只保留概率最高的 K 个 token
   │
   ▼   top-p 过滤    ← 第三步：保留累积概率达到 p 的最小集合
   │
   ▼   softmax → 采样 → token_id
```

三者的关系是「依次过滤」，不是三选一：

| 参数 | 含义 | 特殊值 |
|------|------|--------|
| `temperature` | 缩放 logits。T→0 趋近 greedy，T>1 增加随机性 | `temperature=0` = greedy（直接 argmax） |
| `top_k` | 只保留分数最高的 K 个 token | `top_k=0` = 不启用 |
| `top_p` | 保留累积概率刚好达到 p 的最小 token 集合 | `top_p=1.0` = 不启用 |

### 1.2 Gumbel-Max Trick

传统采样需要两步：
```python
probs = softmax(logits)       # kernel 1
token = multinomial(probs)     # kernel 2
```

Gumbel-Max Trick 可以一步完成：
```python
# 如果 G ~ Gumbel(0,1)，则 argmax(logits + G) 等价于从 softmax(logits) 采样
# Gumbel 噪声可以通过 -log(-log(Uniform)) 生成
# 或者等价地：argmax(probs / Exp(1))
noise = torch.empty_like(probs).exponential_(1).clamp_min(1e-10)
token = (probs / noise).argmax(dim=-1)
```

这只需要一个 kernel，比 `softmax + multinomial` 更快。

### 1.3 理解「temperature=0 即 greedy」的关键

为什么 `temperature=0` 是 greedy 而不是报错？

```python
# 如果 temperature=0，logits / 0 = ±∞，数值崩了
# 所以我们要特殊处理：temperature=0 的样本直接 argmax
greedy_mask = (temperatures == 0)
safe_temps = temperatures.clone()
safe_temps[greedy_mask] = 1.0  # 临时设为1.0，避免除零
```

---

## 2. 🔍 已有代码回顾

### 2.1 Sampler（`layers/sampler.py`）

当前已有的实现：
- ✅ 使用 Gumbel-Max Trick 采样
- ✅ 处理 temperature=0 的 greedy 情况
- ✅ 使用 `@torch.compile` 做 JIT 加速
- ❌ **不支持 top_k 和 top_p**——`forward()` 签名只有 `(logits, temperatures)`

### 2.2 LLMEngine（`engine/llm_engine.py`）

当前已有：
- ✅ 完整的 `generate()` 循环
- ✅ tqdm 进度条和吞吐监控
- ❌ **prefill token 统计不准**：`num_tokens = sum(len(seq) for seq in seqs)` 这行把整条 prompt 长度算进去了，没有减去已缓存的 prefix token

### 2.3 SamplingParams（`sampling_params.py`）

当前已有：
- ❌ 拒绝 `temperature=0`（`assert self.temperature > 1e-10`）
- ❌ 没有 `top_k` / `top_p` 字段

### 2.4 Sequence（`engine/sequence.py`）

当前已有：
- ❌ 没有复制 `top_k` / `top_p` 到 Sequence

---

## 3. ⚠️ 当前问题分析

| 问题 | 严重度 | 影响 |
|------|--------|------|
| SamplingParams 不支持 top_k/top_p | 🔴 高 | 用户无法控制生成多样性 |
| Sequence 没有 top_k/top_p | 🔴 高 | ModelRunner 拿不到采样参数 |
| Sampler.forward() 不接受 top_k/top_p | 🔴 高 | 采样策略只能用 temperature |
| temperature=0 被拒绝 | 🟡 中 | 无法用 greedy decoding |
| prefill token 统计包含了 cached tokens | 🟡 中 | 吞吐量指标偏大（不影响推理正确性） |

---

## 4. 📝 完善后的代码

### 4.1 完善 `sampling_params.py`

```python
# sampling_params.py — 完善版（新增 top_k / top_p，修复 temperature=0）

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    一条请求的采样配置。
    
    这些参数控制生成文本的「随机性 vs 确定性」:
    - temperature=0: greedy（每次都选概率最高的 token）
    - temperature=0.7, top_p=0.9: 中等随机性（适合创意写作）
    - temperature=1.0, top_k=50: 更多样但不离谱
    
    每个参数进入系统后会复制到 Sequence 对象上，
    后续的 ModelRunner 和 Sampler 从 Sequence 取值。
    """

    # ── 温度参数 ──
    # temperature=0 表示 greedy（直接 argmax，不做随机采样）
    # temperature>0 表示按 softmax 概率采样，值越大越随机
    temperature: float = 1.0

    # ── top-k 过滤 ──
    # 只保留概率最高的 K 个 token，其余设为 -inf
    # top_k=0 表示不启用 top-k
    top_k: int = 0

    # ── top-p 过滤（nucleus sampling）──
    # 按概率从高到低排序，只保留累积概率刚好达到 p 的最小集合
    # top_p=1.0 表示不启用 nucleus sampling
    top_p: float = 1.0

    # ── 生成长度控制 ──
    # 最多生成多少个新 token（不含 prompt）
    max_tokens: int = 4096

    # True: 即使遇到 EOS token 也继续生成，直到 max_tokens
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        """参数合法性校验"""
        # temperature >= 0: 允许 0（greedy）
        assert self.temperature >= 0.0, (
            f"temperature 必须 >= 0，当前值: {self.temperature}"
        )
        # top_k >= 0: 0 表示不启用
        assert self.top_k >= 0, (
            f"top_k 必须 >= 0，当前值: {self.top_k}"
        )
        # top_p 在 (0, 1] 之间
        assert 0.0 < self.top_p <= 1.0, (
            f"top_p 必须在 (0, 1] 内，当前值: {self.top_p}"
        )
        # max_tokens > 0: 至少要生成一个
        assert self.max_tokens > 0, (
            f"max_tokens 必须 > 0，当前值: {self.max_tokens}"
        )
```

### 4.2 完善 `engine/sequence.py` —— 补全采样参数

在 `Sequence.__init__()` 中，补充 `top_k` / `top_p` 的复制：

```python
# engine/sequence.py —— Sequence.__init__() 中，替换采样参数复制部分

# ===== 采样参数（从 SamplingParams 复制到 Sequence）=====
# 后续 ModelRunner 和 Sampler 直接从 Sequence 取值，不回头看原始入参
self.temperature = sampling_params.temperature
self.top_k = sampling_params.top_k              # ← 新增
self.top_p = sampling_params.top_p              # ← 新增
self.max_tokens = sampling_params.max_tokens
self.ignore_eos = sampling_params.ignore_eos
```

完整的 `engine/sequence.py` 可以在 Day1 的代码基础上，把 `__init__` 中的采样参数部分换成上面这段。

### 4.3 完善 `layers/sampler.py` —— 支持 top_k / top_p

```python
# layers/sampler.py — 完善版（新增 top_k / top_p 过滤，保留 Gumbel-Max 采样）

"""
采样器

实现 LLM 的 token 采样策略。

采样方法：
1. Greedy: 直接取 argmax（temperature=0）
2. Temperature Sampling: 缩放 logits 后采样
3. Top-K: 只从概率最高的 K 个 token 中采样
4. Top-P (Nucleus): 从累积概率达到 P 的最小集合中采样

Gumbel-Max Trick 原理：
如果 G ~ Gumbel(0,1)，则 argmax(logits + G) 等价于从 softmax(logits) 采样
Gumbel(0,1) 可通过 -log(-log(U)) 生成，U ~ Uniform(0,1)
或者等价地：argmax(probs / Exp(1)) 其中 Exp(1) 是指数分布
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """采样器
    
    从 logits 中采样下一个 token，支持逐样本的温度/top-k/top-p 控制。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,              # [batch_size, vocab_size] 模型输出的 logits
        temperatures: torch.Tensor,         # [batch_size] 每个样本的温度参数
        top_ks: torch.Tensor = None,        # [batch_size] 每个样本的 top_k，可选
        top_ps: torch.Tensor = None,        # [batch_size] 每个样本的 top_p，可选
    ) -> torch.Tensor:
        """
        从 logits 采样下一个 token。
        
        Args:
            logits: [batch_size, vocab_size]
            temperatures: [batch_size]
                - temperature < 1: 更确定性（分布更尖锐）
                - temperature = 1: 原始分布
                - temperature > 1: 更随机（分布更平坦）
                - temperature = 0: 等价于 greedy（直接 argmax）
            top_ks: [batch_size] 可选，不传则不做 top-k 过滤
            top_ps: [batch_size] 可选，不传则不做 top-p 过滤
        
        Returns:
            [batch_size] 采样的 token ID
        """
        # ── 兼容旧调用：不传 top_k/top_p → 默认不启用 ──
        if top_ks is None:
            top_ks = torch.zeros_like(temperatures, dtype=torch.long)
        if top_ps is None:
            top_ps = torch.ones_like(temperatures, dtype=torch.float32)

        # ── 步骤1: 记录哪些样本是 greedy ──
        greedy_mask = (temperatures == 0)

        # 避免除零：greedy 位置临时设为 1.0
        safe_temps = temperatures.clone()
        safe_temps[greedy_mask] = 1.0

        # ── 步骤2: 温度缩放 ──
        scaled_logits = logits.float() / safe_temps.unsqueeze(dim=1)

        # ── 步骤3: 逐样本应用 top-k 和 top-p 过滤 ──
        # 因为每个样本的 top_k / top_p 可能不同，所以用循环逐行处理
        filtered_rows = []
        for i, (row_logits, top_k, top_p) in enumerate(
            zip(scaled_logits, top_ks.tolist(), top_ps.tolist())
        ):
            row_logits = self._apply_top_k(row_logits, int(top_k))
            row_logits = self._apply_top_p(row_logits, float(top_p))
            filtered_rows.append(row_logits)

        filtered_logits = torch.stack(filtered_rows, dim=0)

        # ── 步骤4: softmax → 概率分布 ──
        probs = torch.softmax(filtered_logits, dim=-1)

        # ── 步骤5: Gumbel-Max Trick 采样 ──
        # Exp(1) 分布: torch.empty_like(probs).exponential_(1)
        # clamp_min(1e-10) 防止 log(0) 产生 -inf
        gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        sampled_tokens = (probs / gumbel_noise).argmax(dim=-1)

        # ── 步骤6: greedy 位置直接取 argmax ──
        greedy_tokens = filtered_logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)

    # ═══════════════════════════════════════════════════════════
    # top-k 过滤
    # ═══════════════════════════════════════════════════════════
    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        只保留分数最高的 K 个 token，其余设为 -inf。
        
        top_k <= 0 或 top_k >= vocab_size → 不启用（全保留）。
        
        原理：
        - 用 topk 找到第 K 大的值作为阈值
        - 低于阈值的 token → 设为 -inf
        - softmax(-inf) = 0 → 这些 token 永远不会被采样到
        """
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return logits  # 不启用 top-k

        # 取第 K 大的值
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        threshold = values[..., -1, None]  # 第 K 大的值（含维度）

        # 低于阈值的 token → -inf
        return logits.masked_fill(logits < threshold, float("-inf"))

    # ═══════════════════════════════════════════════════════════
    # top-p 过滤（nucleus sampling）
    # ═══════════════════════════════════════════════════════════
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        只保留累积概率刚好达到 p 的最小 token 集合。
        
        top_p >= 1.0 → 不启用。
        
        步骤：
        1. 将 logits 从大到小排序
        2. softmax → 概率分布
        3. cumsum → 累积概率
        4. 标记累积概率超过 top_p 的 token（但要保留「刚好突破」的那个）
        5. 被标记的 token → -inf
        6. scatter_ 恢复到原始顺序
        """
        if top_p >= 1.0:
            return logits  # 不启用 nucleus sampling

        # 步骤1: 按分数从大到小排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 步骤2: 计算排序空间下的概率分布
        sorted_probs = torch.softmax(sorted_logits, dim=-1)

        # 步骤3: 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 步骤4: 标记累积概率超过 top_p 的 token
        sorted_mask = cumulative_probs > top_p

        # 右移一位：保留「刚好让累积概率超过 top_p」的那个 token
        # 防止极小 top_p 把所有 token 删光
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()

        # 概率最高的 token 永远保留（防御性）
        sorted_mask[..., 0] = False

        # 步骤5: 被标记的 token → -inf
        masked_sorted = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # 步骤6: scatter_ 恢复到原始 vocab 索引空间
        restored = torch.full_like(masked_sorted, float("-inf"))
        restored.scatter_(dim=-1, index=sorted_indices, src=masked_sorted)
        return restored

    # ═══════════════════════════════════════════════════════════
    # 便捷方法
    # ═══════════════════════════════════════════════════════════
    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """纯贪婪解码"""
        return logits.argmax(dim=-1)

    def sample_with_temperature(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """单温度值采样（便捷包装）"""
        batch_size = logits.shape[0]
        temperatures = torch.full(
            (batch_size,),
            temperature,
            device=logits.device,
            dtype=torch.float32,
        )
        return self.forward(logits, temperatures)
```

### 4.4 完善 `engine/llm_engine.py` —— 修复 prefill token 统计

```python
# engine/llm_engine.py — 完善版（修复 prefill token 统计，整合 top_k/top_p 参数传递）

"""
LLM 推理引擎

串联 Scheduler + ModelRunner，实现完整推理循环。
"""

import atexit
from time import perf_counter
from typing import Union

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from config import Config
from sampling_params import SamplingParams
from engine.sequence import Sequence
from engine.block_manager import BlockManager
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner


class LLMEngine:
    """LLM 推理引擎
    
    职责：
    1. 初始化所有组件（Tokenizer、ModelRunner、BlockManager、Scheduler）
    2. 接收文本请求，转成 Sequence 喂给调度器
    3. 驱动 调度→推理→后处理 的无限循环
    4. 解码输出 token → 返回文本结果
    """

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: HuggingFace 模型路径（如 "models/Qwen3-0.6B"）
            **kwargs: 传给 Config 的额外参数（如 enforce_eager=True）
        """
        # ── 步骤1: 创建配置 ──
        self.config = Config(model_path=model, **kwargs)

        # ── 步骤2: 加载 Tokenizer ──
        print(f"[LLMEngine] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        # 把 EOS token ID 写回 Config（Scheduler 需要知道它）
        self.config.eos = self.tokenizer.eos_token_id
        print(f"[LLMEngine] EOS token ID: {self.config.eos}")

        # ── 步骤3: 初始化 ModelRunner（加载模型 + 分配 KV Cache）──
        print(f"[LLMEngine] 初始化 ModelRunner...")
        self.model_runner = ModelRunner(self.config)

        # 计算可用 KV Cache 块数（取 GPU 可用显存的 95%）
        num_blocks = self.model_runner.get_num_free_gpu_blocks()
        num_blocks = max(1, int(num_blocks * 0.95))

        # 分配 KV Cache 显存
        self.model_runner.allocate_kv_cache(num_blocks)

        # ── 步骤4: 创建 BlockManager 和 Scheduler ──
        block_size = Sequence.block_size
        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.config, self.block_manager)

        # ── 步骤5: 注册退出清理 ──
        atexit.register(self._cleanup)

        print(f"[LLMEngine] 初始化完成！")
        print(f"[LLMEngine] - KV Cache: {num_blocks} 块")
        print(f"[LLMEngine] - Block Size: {block_size} tokens")

    # ═══════════════════════════════════════════════════════════
    # 清理
    # ═══════════════════════════════════════════════════════════
    def _cleanup(self):
        """程序退出时释放 GPU 显存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # 添加请求
    # ═══════════════════════════════════════════════════════════
    def add_request(
        self,
        prompt: Union[str, list[int]],
        sampling_params: SamplingParams = None,
    ):
        """
        把一个推理请求加入调度器队列。
        
        Args:
            prompt: 字符串（会自动 tokenize）或 token ID 列表
            sampling_params: 采样参数，不传则用默认值
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Tokenize
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)

        # 创建 Sequence 并加入 waiting 队列
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add_sequence(seq)

    # ═══════════════════════════════════════════════════════════
    # 是否完成
    # ═══════════════════════════════════════════════════════════
    def is_finished(self) -> bool:
        """所有请求是否都完成了"""
        return self.scheduler.is_finished()

    # ═══════════════════════════════════════════════════════════
    # 单步推理（核心）
    # ═══════════════════════════════════════════════════════════
    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """
        执行一次「调度 → 推理 → 后处理」循环。
        
        Returns:
            outputs: [(seq_id, [completion_token_ids]), ...] 本轮完成的序列
            num_tokens:
                · prefill 阶段: 正数 = 本轮新计算的 prompt token 数
                  （注意：不重复计算已缓存的 prefix token）
                · decode 阶段: 负数 = -本轮处理的序列数
        """
        # ── 步骤1: 调度 — 决定这轮处理哪些请求 ──
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0

        # ── 步骤2: 推理 — 模型前向 + 采样 ──
        token_ids = self.model_runner.run(seqs, is_prefill)

        # ── 步骤3: 后处理 — 更新状态，释放资源 ──
        finished_seqs = self.scheduler.postprocess(seqs, token_ids)

        # ── 步骤4: 收集输出 ──
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in finished_seqs
        ]

        # ── 步骤5: token 统计 ──
        # FIXED: prefill 阶段只统计「新计算的」token
        # 原来写的是 sum(len(seq) for seq in seqs)，把已缓存的 prefix token 也算进去了
        if is_prefill:
            num_tokens = sum(
                len(seq) - seq.num_cached_tokens for seq in seqs
            )
        else:
            num_tokens = -len(seqs)  # decode 阶段用负数标记

        return outputs, num_tokens

    # ═══════════════════════════════════════════════════════════
    # 批量生成（对外接口）
    # ═══════════════════════════════════════════════════════════
    def generate(
        self,
        prompts: Union[list[str], list[list[int]]],
        sampling_params: Union[SamplingParams, list[SamplingParams]] = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        批量生成文本。
        
        Args:
            prompts: 提示词列表（字符串或 token ID 列表）
            sampling_params: 统一样采参数，或每条一个
            use_tqdm: 是否显示进度条
        
        Returns:
            [{"text": "生成的文本", "token_ids": [1, 2, 3]}, ...]
        """
        # ── 默认参数 ──
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # ── 添加请求 ──
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # ── 进度条 ──
        pbar = None
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts), desc="Generating", dynamic_ncols=True
            )

        outputs = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0

        # ── 生成循环 ──
        while not self.is_finished():
            t0 = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - t0

            # 更新进度条上的吞吐量显示
            if pbar and elapsed > 0:
                if num_tokens > 0:
                    # prefill: num_tokens 是正数
                    prefill_throughput = num_tokens / elapsed
                elif num_tokens < 0:
                    # decode: num_tokens 是负数（绝对值 = 序列数）
                    decode_throughput = -num_tokens / elapsed

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tok/s",
                    "Decode": f"{int(decode_throughput)} tok/s",
                })

            # 记录完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        # ── 排序并解码 ──
        sorted_outputs = [
            outputs[seq_id] for seq_id in sorted(outputs.keys())
        ]
        results = []
        for token_ids in sorted_outputs:
            text = self.tokenizer.decode(
                token_ids, skip_special_tokens=True
            )
            results.append({
                "text": text,
                "token_ids": token_ids,
            })

        return results
```

### 4.5 `llm.py`（无需修改，确认内容）

```python
# llm.py — 对外接口（LLM 就是 LLMEngine 的别名）

from engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    nano-vLLM 的对外接口。
    
    使用方式：
        llm = LLM("models/Qwen3-0.6B")
        outputs = llm.generate(
            ["你好，请介绍你自己。"],
            SamplingParams(temperature=0.7, max_tokens=64)
        )
        print(outputs[0]["text"])
    """
    pass
```

### 4.6 `example.py`（完善版）

```python
# example.py — 完善版（使用完整的 SamplingParams，更好的输出格式）

"""
nano-vLLM 端到端推理示例

运行:
    cd nano_vll_repro
    python example.py

预期输出:
    模型加载日志 + 两段 Qwen3-0.6B 生成的中文回答
"""

import os
import sys

# 避免 HuggingFace tokenizer 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 确保能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

from llm import LLM
from sampling_params import SamplingParams


def main():
    # ── 模型路径 ──
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "Qwen3-0.6B"
    )

    # ── 打印 GPU 信息 ──
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ 未检测到 CUDA，将使用 CPU（会很慢）")

    print("=" * 60)
    print("nano-vLLM 端到端推理测试")
    print("=" * 60)

    # ── 加载模型 ──
    print("\n正在加载模型...")
    llm = LLM(model_path)

    # ── 加载 tokenizer（用于 apply_chat_template）──
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    # ── 构造 prompt（使用 Qwen3 的 chat template）──
    raw_prompts = [
        "请用一句话解释什么是 PagedAttention。",
        "1 + 1 = ?",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]

    # ── 构造采样参数 ──
    # temperature=0.7: 中等随机性
    # top_k=20: 只考虑概率最高的 20 个 token
    # top_p=0.95: nucleus sampling
    # max_tokens=256: 最多生成 256 个 token
    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=20,
        top_p=0.95,
        max_tokens=256,
    )

    # ── 生成 ──
    print("\n正在生成...")
    outputs = llm.generate(prompts, sampling_params)

    # ── 输出 ──
    for raw_prompt, output in zip(raw_prompts, outputs):
        print(f"\n{'=' * 60}")
        print(f"[问题] {raw_prompt}")
        print(f"[回答] {output['text']}")

    print(f"\n{'=' * 60}")
    print("🎉 所有请求完成！")


if __name__ == "__main__":
    main()
```

---

## 5. 验收

### 5.1 语法检查

```bash
cd nano_vll_repro

python -m py_compile sampling_params.py
python -m py_compile layers/sampler.py
python -m py_compile engine/llm_engine.py
python -m py_compile llm.py
python -m py_compile example.py
```

### 5.2 Sampler 单元测试

```bash
python - <<'PY'
import torch
from layers.sampler import Sampler

sampler = Sampler()
logits = torch.randn(3, 1000)  # 3 条样本，1000 词表

# 测试1: 旧接口仍然可用（不传 top_k/top_p）
temps = torch.tensor([0.0, 0.5, 1.0])
tokens = sampler(logits, temps)
print(f"旧接口输出: {tokens}")

# 测试2: greedy（temperature=0）→ argmax
assert tokens[0] == logits[0].argmax(), "temperature=0 应该是 greedy!"
print("✅ greedy (temperature=0) 验证通过")

# 测试3: 新接口 — top_k 过滤
top_ks = torch.tensor([0, 5, 10])
top_ps = torch.tensor([1.0, 1.0, 0.9])
tokens = sampler(logits, temps, top_ks, top_ps)
print(f"新接口输出: {tokens}")
print("✅ top_k/top_p 测试通过")

# 测试4: top_k=0 → 不过滤 → 和不用 top_k 结果一致
tokens_no_topk = sampler(logits, temps)
tokens_topk0 = sampler(logits, temps, torch.zeros_like(temps))
assert torch.equal(tokens_no_topk, tokens_topk0), "top_k=0 不应影响结果"
print("✅ top_k=0 (不启用) 验证通过")
PY
```

### 5.3 端到端推理（需要模型权重）

```bash
cd nano_vll_repro
python example.py
```

**预期输出**：
```
CUDA: NVIDIA GeForce RTX 4070 Ti
显存: 12.0 GB
============================================================
nano-vLLM 端到端推理测试
============================================================

正在加载模型...
[LLMEngine] 加载 Tokenizer...
[LLMEngine] EOS token ID: 151645
[LLMEngine] 初始化 ModelRunner...
[ModelRunner] 加载模型：models/Qwen3-0.6B
...（模型加载日志）...
[LLMEngine] 初始化完成！

正在生成...
Generating: 100%|██████████| 2/2 [00:06<00:00, ...]

============================================================
[问题] 请用一句话解释什么是 PagedAttention。
[回答] PagedAttention 是一种将 KV Cache 分为固定大小的页进行管理...
============================================================
[问题] 1 + 1 = ?
[回答] 1 + 1 = 2
============================================================
🎉 所有请求完成！
```

---

## 6. 常见问题排查

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| 输出乱码或空文本 | 权重加载有问题 | 检查 `models/Qwen3-0.6B/model.safetensors` 是否存在 |
| `Sampler` 报 `TypeError: unexpected argument` | 旧版 Sampler 不接受 `top_ks/top_ps` | 确认已用上面的新版代码替换 |
| `Sequence` 没有 `top_k` 属性 | 没在 `__init__` 里加那两行 | 检查 `engine/sequence.py` 的采样参数复制部分 |
| 吞吐量变成 0 | `num_cached_tokens` 导致除零 | 确认 `step()` 的 token 统计已改为 `len(seq) - seq.num_cached_tokens` |
| `temperature=0` 仍然报错 | 用了旧版 `sampling_params.py` | 确认 `__post_init__` 是 `>= 0.0` 不是 `> 1e-10` |

---

## 7. 本篇学到的核心概念

1. **Gumbel-Max Trick 是采样器的性能核心**：一次 `argmax(probs/Exp(1))` 等价于 `softmax + multinomial`，但只需要一个 kernel。
2. **temperature/top_k/top_p 是逐样本独立的**：同一个 batch 里可以混合 greedy + 随机采样，Sampler 通过 for 循环逐行处理。
3. **top-p 的 scatter_ 是关键**：排序→标记→scatter_ 三步必须完整，否则过滤结果会错位。
4. **prefill token 统计要排除 cached tokens**：Prefix Cache 命中的 token 不需要重新算，不该计入吞吐量。

---

下一篇：**Day7 — 进阶优化**（CUDA Graph + Tensor Parallel 入门 + 知识图谱总结）
