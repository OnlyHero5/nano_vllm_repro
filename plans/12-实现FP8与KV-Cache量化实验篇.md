# 12. 实现 FP8 与 KV Cache 量化实验篇

这一篇进入另一个很容易被“宣传口号”带偏的方向：`FP8 / KV cache quantization`。

这一篇的目标不是假装当前仓库已经具备完整 FP8 推理后端，而是：

> 站在当前仓库的真实 PyTorch + FlashAttention + PagedAttention 主线上，讲清楚 FP8 权重、激活和 KV cache 量化分别是什么，以及哪些内容适合现在就写成独立实验，哪些内容还不应该硬塞回主线。

这一篇只做四件事：

1. 把 `FP8 weight`、`FP8 activation`、`FP8 KV cache` 的概念边界讲清楚。
2. 新增一个完整可运行的 `fp8_kvcache_proto.py` 独立实验脚本。
3. 在实验脚本里实现 block-wise 量化 / 反量化、KV cache 写入与读取原型。
4. 说明未来如果要接回当前主线，最自然的边界在哪。

这一篇不做下面这些事：

- 不修改当前 `engine/model_runner.py` 的真实 KV cache 分配逻辑。
- 不修改 `layers/attention.py` 的真实 FlashAttention 调用路径。
- 不假装当前仓库已经有 Hopper / Blackwell 上的硬件 FP8 kernel。
- 不把 `torch.float8_e4m3fn` 直接宣称成“当前路径已经能端到端跑通”。
- 不把 weight、activation、KV cache 三种量化路径混成一个开关。

原因很简单：

> “支持 FP8”这句话本身太宽了。对推理系统来说，weight、activation、KV cache 的量化语义、误差来源和工程接入点都不一样。

---

## 1. 先分清三种不同的量化对象

### 1.1 FP8 weight

意思是：模型参数以 FP8 形式存储，前向时再按需要解码 / dequant / cast 参与计算。

它主要影响：

- 模型权重显存占用。
- 参数加载路径。
- GEMM 输入格式。

### 1.2 FP8 activation

意思是：中间激活张量也以 FP8 表示，或在某些 kernel 内部按 FP8 计算。

它主要影响：

- kernel 数值稳定性。
- layernorm / matmul / attention 的算子实现。
- 是否需要 per-tensor / per-channel scale。

### 1.3 FP8 KV cache

意思是：写进 `kv_cache` 的 K/V 张量不再是 `float16` 或 `bfloat16`，而是更低精度表示，并配套 scale 信息。

它主要影响：

- KV cache 显存占用。
- cache 写入与读取协议。
- decode attention 读取 K/V 的方式。

这三者最大的误区就是：

> 很多人会把“模型支持 FP8”误听成“当前整个推理图里所有张量都可以自动换成 FP8”。

这不对。

---

## 2. 当前代码是什么状态

与 KV cache 量化最相关的当前文件有：

1. `engine/model_runner.py`
2. `layers/attention.py`
3. `utils/context.py`
4. `engine/block_manager.py`

### 2.1 当前 `ModelRunner.allocate_kv_cache()` 固定分配 `torch.float16`

当前代码里：

```python
cache = torch.zeros(
    2,
    num_blocks,
    self.block_size,
    self.num_kv_heads,
    self.head_dim,
    dtype=torch.float16,
    device=self.device,
)
```

这意味着：

- 当前 KV cache 协议默认是 fp16。
- 没有额外 scale buffer。
- 写入和读取路径都默认“直接存真值”。

### 2.2 当前 `layers/attention.py` 的 `store_kvcache()` 也假设直接写半精度张量

它做的是：

- 根据 `slot_mapping` 直接把 K/V 写入 `kv_cache`。
- decode 时又直接把 `k_cache / v_cache` 交给 flash-attn。

所以当前真实主线的 KV cache 协议是“无量化、直接写入”。

---

## 3. 当前教学仓库里最适合的实验边界

这一篇采用下面这条边界：

1. 不改主线 KV cache tensor。
2. 新增独立实验文件 `utils/fp8_kvcache_proto.py`。
3. 在这个实验文件里完整实现 block-wise 量化、block-wise 反量化、K/V 写入量化 cache、读取量化 cache 并恢复近似值。
4. 测试也写成独立实验测试，不污染现有主线测试。

为什么选 block-wise：

- 当前仓库的 KV cache 本来就是 block 组织。
- 社区里很多低精度 KV cache 方案也会按 block / page / chunk 搭配 scale。
- block-wise 比 per-token 和 per-head 单位更容易贴合当前 `BlockManager` 语义。

---

## 4. 新增 `utils/fp8_kvcache_proto.py`

下面给出完整教学原型。这个文件不依赖当前仓库不存在的自定义 kernel，直接用 PyTorch 张量把“低精度 KV cache 存取协议”讲清楚。

```python
"""教学版 FP8 / block-wise KV cache 量化原型。

这个文件的重点是量化协议，不是最终高性能 kernel。
为了在普通 PyTorch 环境下稳定运行，教学版先使用 int8 容器模拟低精度 cache。
"""

from dataclasses import dataclass

import torch


@dataclass
class QuantizedKVCache:
    """
    教学版低精度 KV cache。

    qk_cache / qv_cache 保存量化后的 K/V 数据。
    k_scales / v_scales 按 block 和 KV head 记录缩放因子。
    """
    qk_cache: torch.Tensor
    qv_cache: torch.Tensor
    k_scales: torch.Tensor
    v_scales: torch.Tensor
    block_size: int


class FP8KVCacheProto:
    """
    教学版 block-wise KV cache 量化器。
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

        qk_cache = torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.int8,
            device=device,
        )
        qv_cache = torch.zeros_like(qk_cache)
        k_scales = torch.ones(num_blocks, num_kv_heads, 1, dtype=torch.float32, device=device)
        v_scales = torch.ones(num_blocks, num_kv_heads, 1, dtype=torch.float32, device=device)

        self.cache = QuantizedKVCache(
            qk_cache=qk_cache,
            qv_cache=qv_cache,
            k_scales=k_scales,
            v_scales=v_scales,
            block_size=block_size,
        )

    def _quantize_block(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对一个 block 的 K 或 V 做 block-wise 量化。

        输入形状：
        - [block_size, num_kv_heads, head_dim]

        返回：
        - quantized: int8 量化结果。
        - scales: [num_kv_heads, 1]。
        """
        assert x.ndim == 3, "输入必须是 [block_size, num_kv_heads, head_dim]"

        max_abs = x.abs().amax(dim=(0, 2), keepdim=False).unsqueeze(-1)
        max_abs = torch.clamp(max_abs, min=1e-6)
        scales = max_abs / 127.0
        quantized = torch.clamp(torch.round(x / scales.unsqueeze(0)), -127, 127).to(torch.int8)
        return quantized, scales

    def _dequantize_block(self, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        把一个量化 block 恢复成近似浮点值。
        """
        return q.float() * scales.unsqueeze(0)

    def write_block(self, block_id: int, k_block: torch.Tensor, v_block: torch.Tensor) -> None:
        """
        把一个完整 block 的 K/V 写入量化 cache。
        """
        assert 0 <= block_id < self.num_blocks, "block_id 越界"
        assert k_block.shape == (self.block_size, self.num_kv_heads, self.head_dim)
        assert v_block.shape == (self.block_size, self.num_kv_heads, self.head_dim)

        qk, k_scales = self._quantize_block(k_block)
        qv, v_scales = self._quantize_block(v_block)

        self.cache.qk_cache[block_id].copy_(qk)
        self.cache.qv_cache[block_id].copy_(qv)
        self.cache.k_scales[block_id].copy_(k_scales)
        self.cache.v_scales[block_id].copy_(v_scales)

    def read_block(self, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从量化 cache 读取一个 block，并恢复成近似浮点值。
        """
        assert 0 <= block_id < self.num_blocks, "block_id 越界"

        qk = self.cache.qk_cache[block_id]
        qv = self.cache.qv_cache[block_id]
        k_scales = self.cache.k_scales[block_id]
        v_scales = self.cache.v_scales[block_id]

        k_block = self._dequantize_block(qk, k_scales)
        v_block = self._dequantize_block(qv, v_scales)
        return k_block, v_block

    def get_memory_report(self) -> dict:
        """
        导出一份教学版显存统计。
        """
        num_elements = self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim
        fp16_bytes = 2 * 2 * num_elements
        int8_cache_bytes = 2 * 1 * num_elements
        scales_bytes = 2 * self.num_blocks * self.num_kv_heads * 4
        quantized_total = int8_cache_bytes + scales_bytes

        return {
            "fp16_kv_cache_bytes": fp16_bytes,
            "quantized_kv_cache_bytes": quantized_total,
            "compression_ratio": fp16_bytes / quantized_total if quantized_total > 0 else 0.0,
        }
```

### 4.1 为什么这里先用 `int8` 容器，而不是直接写 `float8`

这篇的目标是：

> 在普通 PyTorch 环境里，完整讲清楚“块级 scale + 量化 cache 协议”。

如果一开始就把教学原型强绑定到某个特定 GPU 架构、`torch.float8_*` 格式或 FlashAttention 分支，读者会很难分清哪些是协议本身，哪些是硬件后端细节。

---

## 5. 新增 `tests/test_Day12_fp8_kvcache_proto.py`

这份测试只验证量化协议，不依赖主线模型。

```python
"""Day12 FP8 / 低精度 KV cache 原型测试。"""

import sys

sys.path.insert(0, ".")

import torch

from utils.fp8_kvcache_proto import FP8KVCacheProto


def test_quantized_kvcache_shapes():
    proto = FP8KVCacheProto(
        block_size=4,
        num_blocks=8,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
    )

    assert proto.cache.qk_cache.shape == (8, 4, 2, 8)
    assert proto.cache.qv_cache.shape == (8, 4, 2, 8)
    assert proto.cache.k_scales.shape == (8, 2, 1)
    assert proto.cache.v_scales.shape == (8, 2, 1)


def test_write_then_read_block_restores_close_values():
    proto = FP8KVCacheProto(
        block_size=4,
        num_blocks=8,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
    )

    torch.manual_seed(0)
    k = torch.randn(4, 2, 8) * 0.1
    v = torch.randn(4, 2, 8) * 0.1

    proto.write_block(0, k, v)
    restored_k, restored_v = proto.read_block(0)

    assert restored_k.shape == k.shape
    assert restored_v.shape == v.shape
    assert torch.allclose(restored_k, k, atol=0.02, rtol=0.2)
    assert torch.allclose(restored_v, v, atol=0.02, rtol=0.2)


def test_memory_report_has_compression_ratio():
    proto = FP8KVCacheProto(
        block_size=4,
        num_blocks=8,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
    )
    report = proto.get_memory_report()

    assert report["fp16_kv_cache_bytes"] > 0
    assert report["quantized_kv_cache_bytes"] > 0
    assert report["compression_ratio"] > 1.0
```

---

## 6. 如果以后真要接回当前主线

建议按下面顺序走：

### 6.1 先扩 `Config`

例如：

- `kv_cache_dtype = "float16" | "int8" | "fp8"`
- `kv_cache_quant_scheme = "per_block_head"`

这样主线至少能明确区分普通 fp16 cache 和低精度 cache 原型路径。

### 6.2 再改 `ModelRunner.allocate_kv_cache()`

当前它只会分配 `list[torch.Tensor]` 每层真实 K/V cache。如果以后接入低精度 cache，就要改成：

- 普通路径：继续分配 fp16 tensor。
- 量化路径：分配量化 cache + scale buffer。

### 6.3 然后改 `layers/attention.py`

当前 `store_kvcache()` 和 decode 路径都假设 cache 里存的就是直接可读的半精度 K/V。如果接入量化 cache，就必须改成写入时先量化、读取时先恢复近似浮点值。

### 6.4 最后才讨论真 FP8 硬件算子

教学版现在还不需要直接碰 Hopper / Blackwell 上的特定 FP8 kernel、FlashAttention 的硬件特化分支或 Triton 自定义 FP8 store / load kernel。

---

## 7. 验收命令

```bash
python -m py_compile utils/fp8_kvcache_proto.py tests/test_Day12_fp8_kvcache_proto.py
python tests/test_Day12_fp8_kvcache_proto.py
```

如果只想快速看量化误差与显存报告，可以再跑：

```bash
python - <<'PY'
import torch
from utils.fp8_kvcache_proto import FP8KVCacheProto

proto = FP8KVCacheProto(block_size=4, num_blocks=8, num_kv_heads=2, head_dim=8, device=torch.device("cpu"))
k = torch.randn(4, 2, 8) * 0.1
v = torch.randn(4, 2, 8) * 0.1
proto.write_block(0, k, v)
restored_k, restored_v = proto.read_block(0)
print("mean abs error k:", (restored_k - k).abs().mean().item())
print("memory report:", proto.get_memory_report())
PY
```

---

## 8. 常见坑

1. **把 weight、activation、KV cache 三种量化路径混成一个开关。**
   这会让你根本不知道误差和收益来自哪一层。
2. **一上来就把低精度 cache 强接进 `layers/attention.py`。**
   当前主线还没准备好承接这条复杂度。
3. **以为量化 cache 只要换 dtype 就行，不需要 scale。**
   没有 scale，就没有可控的数值恢复路径。
4. **把教学原型写成硬件绑定代码。**
   教学版的目标是讲清协议，不是先锁死某张 GPU。
5. **看到社区仓库写 FP8，就误以为当前所有张量路径都已经能直接换成 FP8。**
   真实工程里，weight、activation、KV cache 的接入点完全不同。

---

## 9. 本篇结束后你应该明白

这一篇最重要的不是“会写一个量化函数”。

真正要学会的是：

1. `FP8 weight`、`FP8 activation`、`FP8 KV cache` 是三条不同的工程路径。
2. 对当前教学仓库来说，最自然先做实验的是“块级低精度 KV cache 协议”。
3. block-wise scale 是把当前 `BlockManager` 语义延续到量化 cache 的关键。
4. 在协议没讲清楚之前，不应该直接把低精度 cache 硬并回 Attention 主线。

下一篇进入 GPU offload 与跨后端扩展总览：

- `13-实现GPU-Offload与跨后端扩展总览.md`
