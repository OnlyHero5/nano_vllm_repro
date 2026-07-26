# Day 12 — KV Cache 量化（int8 模拟，FP8 认知篇）

> **本篇边界**：这里落地的是 KV cache 的 **int8 对称量化模拟**（`int8_sim`，以及用 int8 容器模拟更粗粒度取值的 `pseudo_fp8_sim`）。全篇**不涉及** `torch.float8_e4m3fn` 等 float8 dtype、`torch._scaled_mm`、Hopper FP8 GEMM 或 FlashAttention 的 FP8 cache kernel——真 FP8 只做原理讲解。要讲透的是"低精度容器 + scale + 写前量化 / 读前反量化"这套协议边界，它换成任何精度都成立。
>
> **前置依赖**：本篇修改 `config.py / engine/model_runner.py / utils/context.py / layers/attention.py`，以主线 Day1-Day6 落地后的代码为基础。

KV cache 是长上下文推理的显存大户。当前主线用 fp16 存每个 token 的 K 和 V——如果改成 int8 甚至模拟 fp8，显存直接砍半或更多。代价是精度损失，但 KV cache 对精度的容忍度比权重高得多。

这次加一条**可选**的量化路径：

1. 默认仍然是 fp16 KV cache，行为不变。
2. 新增 `int8_sim` / `pseudo_fp8_sim` 两种量化模拟。
3. 量化路径只服务教学：讲清楚 cache 容器、scale buffer、写入量化、读取反量化、FlashAttention 前恢复半精度这些接口边界。

生产级 FP8 kernel / Hopper FP8 GEMM / FlashAttention FP8 cache kernel 都不做。

`int8_sim` 和 `pseudo_fp8_sim` 默认关闭，只有显式设置才会进入量化路径：

```python
kv_cache_dtype = "quantized"
kv_cache_quant_scheme = "int8_sim"        # 或 "pseudo_fp8_sim"
```

---

## 1. 要改什么

KV cache 量化接入主线，涉及六个接触点：

1. `config.py`：明确开关，默认保持 fp16。
2. `utils/kvcache_quant.py`：量化协议——container 选择、scale 计算、quantize/dequantize、教学版 attention hook。
3. `engine/model_runner.py`：`allocate_kv_cache()` 不再只分配裸 `torch.float16` tensor；量化路径需要每层 cache 和 scale buffers 成对保存。
4. `utils/context.py`：`Context.kv_cache` 类型要能表达“裸 fp16 tensor”和“携带量化元数据的 view”。
5. `layers/attention.py`：写 cache 前可以量化，读 cache 喂给 FlashAttention 前必须反量化到 fp16/bf16。
6. `tests/test_Day12_kvcache_quant.py`：不依赖真实大模型，只验证量化协议、默认路径和 hook 形状语义。

注意：这不是 weight quantization，也不是 activation quantization。只处理 **KV cache storage**。

---

## 2. 文件级改动清单

涉及六个文件：

1. 修改 `config.py`
2. 新增 `utils/kvcache_quant.py`
3. 修改 `engine/model_runner.py`
4. 修改 `utils/context.py`
5. 修改 `layers/attention.py`
6. 新增 `tests/test_Day12_kvcache_quant.py`

`kv_cache_dtype="fp16"` + `kv_cache_quant_scheme="none"` 是默认组合，保持当前行为。

---

## 3. 修改 `config.py`：增加 KV cache 量化开关

### 3.1 在 `Config` dataclass 里增加字段

把字段放在当前 PagedAttention 参数附近：

```python
# PagedAttention 参数
kvcache_block_size: int = 256   # KV cache 块大小
num_kvcache_blocks: int = -1    # KV cache 块数量

# Day12: optional KV cache quantization
# 默认组合保持当前 fp16 KV cache 行为。
# kv_cache_dtype:
# - "fp16": 当前默认路径，cache 直接存半精度 K/V
# - "quantized": 教学量化路径，cache 用低精度容器 + scale buffers
kv_cache_dtype: str = "fp16"

# kv_cache_quant_scheme:
# - "none": 不量化，只能搭配 kv_cache_dtype="fp16"
# - "int8_sim": int8 容器的对称量化模拟
# - "pseudo_fp8_sim": int8 容器模拟 FP8 风格的更粗粒度取值
kv_cache_quant_scheme: str = "none"
```

### 3.2 在 `__post_init__()` 里增加校验

把下面校验放在 `kvcache_block_size` 校验之后、加载 Hugging Face config 之前：

```python
# Day12: KV cache quantization 参数校验
valid_kv_cache_dtypes = {"fp16", "quantized"}
assert self.kv_cache_dtype in valid_kv_cache_dtypes, (
    f"kv_cache_dtype 必须是 {valid_kv_cache_dtypes}，当前为 {self.kv_cache_dtype}"
)

valid_kv_cache_quant_schemes = {"none", "int8_sim", "pseudo_fp8_sim"}
assert self.kv_cache_quant_scheme in valid_kv_cache_quant_schemes, (
    f"kv_cache_quant_scheme 必须是 {valid_kv_cache_quant_schemes}，"
    f"当前为 {self.kv_cache_quant_scheme}"
)

if self.kv_cache_dtype == "fp16":
    assert self.kv_cache_quant_scheme == "none", (
        "kv_cache_dtype='fp16' 时必须使用 kv_cache_quant_scheme='none'，"
        "这样默认路径才保持原始 fp16 KV cache 行为"
    )
else:
    assert self.kv_cache_quant_scheme in {"int8_sim", "pseudo_fp8_sim"}, (
        "kv_cache_dtype='quantized' 时必须选择 int8_sim 或 pseudo_fp8_sim"
    )
```

这个配置设计故意不把 weight、activation、KV cache 混成一个开关。Day12 只管理 KV cache storage。

---

## 4. 新增 `utils/kvcache_quant.py`：教学版量化工具

新建 `utils/kvcache_quant.py`，写入下面完整文件。它只依赖 PyTorch，不依赖真实模型权重，不依赖 FlashAttention，也不依赖 GPU。

```python
"""Day12 optional KV cache quantization utilities.

本文件实现当前 nano-vLLM 学习仓库可选主线使用的 KV cache 量化教学路径。
它支持两种模拟方案：
- int8_sim: int8 容器 + per-block/per-KV-head scale
- pseudo_fp8_sim: int8 容器模拟 FP8 风格的更粗粒度取值

本文件不提供生产级 FP8 kernel，不实现 Hopper FP8 GEMM，也不修改
FlashAttention 内部 kernel。量化 cache 在喂给 FlashAttention 前会反量化为
fp16/bf16 可消费的普通 tensor。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


QUANTIZED_SCHEMES = {"int8_sim", "pseudo_fp8_sim"}


@dataclass(frozen=True)
class KVCacheQuantSpec:
    """KV cache 量化方案描述。

    storage_dtype 是实际 cache container 的 dtype。
    qmin/qmax 是对称量化后的整数范围。
    """

    name: str
    storage_dtype: torch.dtype
    qmin: int
    qmax: int
    scale_dtype: torch.dtype = torch.float32


@dataclass
class KVCacheView:
    """一层 KV cache 的统一视图。

    cache:
        Shape 为 [2, num_blocks, block_size, num_kv_heads, head_dim]。
        cache[0] 是 K cache，cache[1] 是 V cache。

    scales:
        None 表示未量化路径。
        量化路径下 shape 为 [2, num_blocks, num_kv_heads, 1]。
        scales[0] 是 K scale，scales[1] 是 V scale。
    """

    cache: torch.Tensor
    block_size: int
    quant_scheme: str = "none"
    scales: torch.Tensor | None = None

    @property
    def is_quantized(self) -> bool:
        return self.quant_scheme in QUANTIZED_SCHEMES

    @property
    def k_cache(self) -> torch.Tensor:
        return self.cache[0]

    @property
    def v_cache(self) -> torch.Tensor:
        return self.cache[1]

    @property
    def k_scales(self) -> torch.Tensor:
        if self.scales is None:
            raise ValueError("未启用 KV cache 量化时没有 k_scales")
        return self.scales[0]

    @property
    def v_scales(self) -> torch.Tensor:
        if self.scales is None:
            raise ValueError("未启用 KV cache 量化时没有 v_scales")
        return self.scales[1]

    @property
    def nbytes(self) -> int:
        cache_bytes = self.cache.numel() * self.cache.element_size()
        scale_bytes = 0 if self.scales is None else self.scales.numel() * self.scales.element_size()
        return cache_bytes + scale_bytes


def get_quant_spec(quant_scheme: str) -> KVCacheQuantSpec:
    """返回量化方案对应的 container 和整数范围。"""

    if quant_scheme == "int8_sim":
        return KVCacheQuantSpec(
            name="int8_sim",
            storage_dtype=torch.int8,
            qmin=-127,
            qmax=127,
        )

    if quant_scheme == "pseudo_fp8_sim":
        # 教学模拟：仍使用 int8 container，但减少可用整数范围，并在量化时
        # 对较大幅值做更粗粒度 snapping，用来表达 FP8 风格的低精度存储。
        return KVCacheQuantSpec(
            name="pseudo_fp8_sim",
            storage_dtype=torch.int8,
            qmin=-120,
            qmax=120,
        )

    raise ValueError(f"不支持的 KV cache 量化方案：{quant_scheme}")


def select_cache_container(
    kv_cache_dtype: str,
    kv_cache_quant_scheme: str,
) -> tuple[torch.dtype, KVCacheQuantSpec | None]:
    """根据 Config 字段选择 cache container。"""

    if kv_cache_dtype == "fp16":
        if kv_cache_quant_scheme != "none":
            raise ValueError("fp16 KV cache 必须搭配 kv_cache_quant_scheme='none'")
        return torch.float16, None

    if kv_cache_dtype == "quantized":
        return get_quant_spec(kv_cache_quant_scheme).storage_dtype, get_quant_spec(kv_cache_quant_scheme)

    raise ValueError(f"不支持的 kv_cache_dtype：{kv_cache_dtype}")


def allocate_kvcache_view(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    kv_cache_dtype: str,
    kv_cache_quant_scheme: str,
) -> KVCacheView:
    """分配一层 KV cache。

    默认 fp16 路径：
    - cache dtype 为 torch.float16
    - scales 为 None

    量化路径：
    - cache dtype 由量化方案选择，教学版为 torch.int8
    - scales dtype 为 torch.float32
    - 每层 cache 和 scales 保存在同一个 KVCacheView 中
    """

    storage_dtype, quant_spec = select_cache_container(
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_scheme=kv_cache_quant_scheme,
    )

    cache = torch.zeros(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=storage_dtype,
        device=device,
    )

    scales = None
    quant_scheme = "none"
    if quant_spec is not None:
        scales = torch.ones(
            2,
            num_blocks,
            num_kv_heads,
            1,
            dtype=quant_spec.scale_dtype,
            device=device,
        )
        quant_scheme = quant_spec.name

    return KVCacheView(
        cache=cache,
        block_size=block_size,
        quant_scheme=quant_scheme,
        scales=scales,
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def estimate_kvcache_bytes_per_layer(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: str,
    kv_cache_quant_scheme: str,
) -> int:
    """估算一层 KV cache 的字节数。"""

    storage_dtype, quant_spec = select_cache_container(
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_scheme=kv_cache_quant_scheme,
    )
    cache_bytes = 2 * num_blocks * block_size * num_kv_heads * head_dim * _dtype_nbytes(storage_dtype)
    if quant_spec is None:
        return cache_bytes

    scale_bytes = 2 * num_blocks * num_kv_heads * 1 * _dtype_nbytes(quant_spec.scale_dtype)
    return cache_bytes + scale_bytes


def _snap_to_pseudo_fp8_levels(q: torch.Tensor) -> torch.Tensor:
    """把整数值吸附到更粗粒度的取值点。

    这不是 IEEE FP8 编码器，只是教学版 pseudo_fp8_sim 的误差模拟。
    """

    abs_q = q.abs()
    step = torch.where(
        abs_q < 16,
        torch.ones_like(abs_q),
        torch.where(abs_q < 64, torch.full_like(abs_q, 2.0), torch.full_like(abs_q, 4.0)),
    )
    return q.sign() * torch.round(abs_q / step) * step


def quantize_block_head(
    x: torch.Tensor,
    quant_scheme: str,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对一个 block 做 per-KV-head 量化。

    Args:
        x: [block_size, num_kv_heads, head_dim]
        quant_scheme: "int8_sim" 或 "pseudo_fp8_sim"
        eps: 防止全 0 block 得到 0 scale

    Returns:
        q: 量化后的 cache block，shape 与 x 相同
        scales: [num_kv_heads, 1]
    """

    if x.ndim != 3:
        raise ValueError("x 必须是 [block_size, num_kv_heads, head_dim]")

    spec = get_quant_spec(quant_scheme)
    x_float = x.float()
    max_abs = x_float.abs().amax(dim=(0, 2), keepdim=False).clamp_min(eps).unsqueeze(-1)
    scales = max_abs / float(spec.qmax)

    q = torch.round(x_float / scales.unsqueeze(0))
    if quant_scheme == "pseudo_fp8_sim":
        q = _snap_to_pseudo_fp8_levels(q)

    q = torch.clamp(q, spec.qmin, spec.qmax).to(spec.storage_dtype)
    return q, scales.to(spec.scale_dtype)


def dequantize_block_head(
    q: torch.Tensor,
    scales: torch.Tensor,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """把一个量化 block 恢复为 FlashAttention 可消费的浮点 tensor。"""

    return (q.float() * scales.float().unsqueeze(0)).to(target_dtype)


def _slot_to_block_offset(slot: int, block_size: int) -> tuple[int, int]:
    return int(slot) // block_size, int(slot) % block_size


def store_quantized_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: KVCacheView,
    slot_mapping: torch.Tensor,
) -> None:
    """教学版量化 store hook。

    当前 attention 产生的是浮点 K/V。量化路径在写入 cache 前做：
    1. 根据 slot_mapping 找到 block_id / block_offset。
    2. 反量化当前 block，保留同 block 里已有 token。
    3. 写入当前 token 的 K/V。
    4. 对整个 block 重新计算 scale 并量化回低精度 container。

    这个实现强调协议正确性，不追求 kernel 性能。
    """

    if not kv_cache.is_quantized:
        raise ValueError("store_quantized_kv_cache 只接受量化 KVCacheView")

    if k.shape != v.shape:
        raise ValueError(f"k/v shape 必须一致，当前 k={k.shape}, v={v.shape}")
    if k.ndim != 3:
        raise ValueError("k/v 必须是 [num_tokens, num_kv_heads, head_dim]")
    if slot_mapping.numel() != k.shape[0]:
        raise ValueError("slot_mapping 长度必须等于 num_tokens")

    slots = slot_mapping.detach().to("cpu").tolist()
    num_blocks = kv_cache.cache.shape[1]

    for token_idx, slot in enumerate(slots):
        block_id, block_offset = _slot_to_block_offset(slot, kv_cache.block_size)
        if not (0 <= block_id < num_blocks):
            raise IndexError(f"slot={slot} 对应的 block_id={block_id} 越界")

        k_block = dequantize_block_head(
            kv_cache.k_cache[block_id],
            kv_cache.k_scales[block_id],
            target_dtype=torch.float32,
        )
        v_block = dequantize_block_head(
            kv_cache.v_cache[block_id],
            kv_cache.v_scales[block_id],
            target_dtype=torch.float32,
        )

        k_block[block_offset].copy_(k[token_idx].to(device=kv_cache.cache.device, dtype=torch.float32))
        v_block[block_offset].copy_(v[token_idx].to(device=kv_cache.cache.device, dtype=torch.float32))

        qk, k_scales = quantize_block_head(k_block, kv_cache.quant_scheme)
        qv, v_scales = quantize_block_head(v_block, kv_cache.quant_scheme)

        kv_cache.k_cache[block_id].copy_(qk)
        kv_cache.v_cache[block_id].copy_(qv)
        kv_cache.k_scales[block_id].copy_(k_scales)
        kv_cache.v_scales[block_id].copy_(v_scales)


def store_kv_cache_for_attention(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: KVCacheView,
    slot_mapping: torch.Tensor,
) -> None:
    """attention 可调用的统一 store hook。

    未量化路径直接写 fp16 cache；量化路径调用 store_quantized_kv_cache。
    """

    if kv_cache.is_quantized:
        store_quantized_kv_cache(k, v, kv_cache, slot_mapping)
        return

    slots = slot_mapping.detach().to("cpu").tolist()
    num_blocks = kv_cache.cache.shape[1]

    for token_idx, slot in enumerate(slots):
        block_id, block_offset = _slot_to_block_offset(slot, kv_cache.block_size)
        if not (0 <= block_id < num_blocks):
            raise IndexError(f"slot={slot} 对应的 block_id={block_id} 越界")
        kv_cache.cache[0, block_id, block_offset].copy_(k[token_idx].to(kv_cache.cache.dtype))
        kv_cache.cache[1, block_id, block_offset].copy_(v[token_idx].to(kv_cache.cache.dtype))


def get_kv_cache_for_attention(
    kv_cache: KVCacheView | torch.Tensor,
    target_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 FlashAttention 可消费的 k_cache / v_cache。

    - 裸 torch.Tensor：保持当前主线路径，直接返回 cache[0] / cache[1]。
    - 未量化 KVCacheView：直接返回内部 fp16 cache，不复制。
    - 量化 KVCacheView：使用 scale 反量化整层 cache，返回 target_dtype tensor。
    """

    if isinstance(kv_cache, torch.Tensor):
        return kv_cache[0], kv_cache[1]

    if not kv_cache.is_quantized:
        return kv_cache.k_cache, kv_cache.v_cache

    k_cache = (kv_cache.k_cache.float() * kv_cache.k_scales[:, None, :, :].float()).to(target_dtype)
    v_cache = (kv_cache.v_cache.float() * kv_cache.v_scales[:, None, :, :].float()).to(target_dtype)
    return k_cache, v_cache
```

### 4.1 两种模拟方案的含义

- `int8_sim`：标准对称 int8 量化模拟，按 block + KV head 维护 scale。
- `pseudo_fp8_sim`：仍使用 int8 container，但缩小整数范围并做粗粒度 snapping，用来观察“更低有效精度”对 cache roundtrip 的影响。

它们都不是生产级 FP8 kernel 支持。FlashAttention 路径仍然吃 `fp16` / `bf16` tensor。

---

## 5. 修改 `engine/model_runner.py`：分配量化 cache 与 scale buffers

### 5.1 增加 import 和类型标注

在 `engine/model_runner.py` 顶部增加：

```python
from utils.kvcache_quant import (
    KVCacheView,
    allocate_kvcache_view,
    estimate_kvcache_bytes_per_layer,
)
```

把 `self.kv_cache` 的类型标注改成：

```python
# KV cache
self.kv_cache: Optional[list[torch.Tensor | KVCacheView]] = None
```

### 5.2 替换 `allocate_kv_cache()`

量化路径下，每层都分配一个 `KVCacheView`：

- `view.cache` 保存 K/V 的低精度 container。
- `view.scales` 保存 K/V 对应的 scale buffers。
- `self.kv_cache[layer_idx]` 就是一层 cache 与 scale 的绑定对象。

默认 `fp16` 路径继续把裸 `torch.Tensor` 放进 `self.kv_cache`，这样原行为保持不变。

```python
def allocate_kv_cache(self, num_blocks: int):
    """预分配 KV Cache 显存。

    默认路径：
    - self.config.kv_cache_dtype == "fp16"
    - self.kv_cache 是 list[torch.Tensor]
    - 每层 tensor shape 为 [2, num_blocks, block_size, num_kv_heads, head_dim]

    量化路径：
    - self.config.kv_cache_dtype == "quantized"
    - self.kv_cache 是 list[KVCacheView]
    - 每层 KVCacheView.cache 保存量化 K/V
    - 每层 KVCacheView.scales 保存 K/V scale buffers
    """

    bytes_per_layer = estimate_kvcache_bytes_per_layer(
        num_blocks=num_blocks,
        block_size=self.block_size,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        kv_cache_dtype=self.config.kv_cache_dtype,
        kv_cache_quant_scheme=self.config.kv_cache_quant_scheme,
    )
    total_bytes = self.num_layers * bytes_per_layer
    print(f"[ModelRunner] KV Cache 显存需求：{total_bytes / 1024**3:.2f} GB")

    self.kv_cache = []
    for _ in range(self.num_layers):
        layer_cache = allocate_kvcache_view(
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.device,
            kv_cache_dtype=self.config.kv_cache_dtype,
            kv_cache_quant_scheme=self.config.kv_cache_quant_scheme,
        )

        if layer_cache.is_quantized:
            self.kv_cache.append(layer_cache)
        else:
            # 保持当前主线的 list[torch.Tensor] 形态，默认路径不变。
            self.kv_cache.append(layer_cache.cache)

    if self.config.kv_cache_dtype == "fp16":
        print(f"[ModelRunner] fp16 KV Cache 分配完成：{num_blocks} 块 × {self.num_layers} 层")
    else:
        print(
            "[ModelRunner] quantized KV Cache 分配完成："
            f"{num_blocks} 块 × {self.num_layers} 层，"
            f"scheme={self.config.kv_cache_quant_scheme}"
        )
```

### 5.3 同步更新可分配块数估算

`get_num_free_gpu_blocks()` 当前按 fp16 每元素 2 bytes 估算。加入量化路径后，用同一个 helper 估算单块单层字节数：

```python
bytes_per_block_per_layer = estimate_kvcache_bytes_per_layer(
    num_blocks=1,
    block_size=self.block_size,
    num_kv_heads=self.num_kv_heads,
    head_dim=self.head_dim,
    kv_cache_dtype=self.config.kv_cache_dtype,
    kv_cache_quant_scheme=self.config.kv_cache_quant_scheme,
)
bytes_per_block = bytes_per_block_per_layer * self.num_layers
num_blocks = int(available_memory // bytes_per_block)
```

这样 `int8_sim` / `pseudo_fp8_sim` 的 cache container 和 scale buffers 都会进入容量估算。

---

## 6. 修改 `utils/context.py`：让 Context 携带量化元数据

`Context.kv_cache` 当前只标注为 `Optional[list[torch.Tensor]]`。加入 optional quantization path 后，它需要能携带 `KVCacheView`。

在 `utils/context.py` 增加 import：

```python
from utils.kvcache_quant import KVCacheView
```

把 `Context` dataclass 里的 KV cache 字段改成：

```python
@dataclass
class Context:
    """
    全局推理上下文。

    Attention 层通过 Context 读取 PagedAttention 所需元数据。
    Day12 之后，kv_cache 可以是：
    - list[torch.Tensor]：默认 fp16 主线路径
    - list[KVCacheView]：optional quantized KV cache 路径
    """

    # ===阶段标识===
    is_prefill: bool = False

    # ===Prefill阶段参数（FlashAttention varlen API 需要）===
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # ===KV Cache 写入参数===
    slot_mapping: torch.Tensor | None = None

    # ===Decode阶段参数===
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int = None
    max_num_blocks: int = None

    # ===KV Cache 引用===
    kv_cache: Optional[list[torch.Tensor | KVCacheView]] = None
```

`ModelRunner.prepare_prefill()` 和 `ModelRunner.prepare_decode()` 仍然把 `self.kv_cache` 传进 `Context`，不需要改变调用位置。

---

## 7. 修改 `layers/attention.py`：store 前量化，FlashAttention 前反量化

这一层是 Day12 的关键：

- 写 cache：attention 产生的 `k` / `v` 仍是浮点 tensor；量化路径在写入 cache 前调用 `store_kv_cache_for_attention()`。
- 读 cache：FlashAttention decode 仍需要半精度 K/V；量化路径通过 `get_kv_cache_for_attention()` 得到反量化后的临时 `k_cache` / `v_cache`。
- 这不是生产级 FP8 kernel；它没有让 FlashAttention 直接读取 FP8 cache。

### 7.1 增加 import

在 `layers/attention.py` 顶部增加：

```python
from utils.kvcache_quant import (
    KVCacheView,
    get_kv_cache_for_attention,
    store_kv_cache_for_attention,
)
```

### 7.2 替换 `store_kvcache()` 的入口逻辑

保留当前 Triton `store_kvcache_kernel` 给默认 fp16 路径使用；只在传入 `KVCacheView` 且启用量化时走教学版量化 hook。

```python
def store_kvcache(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor | KVCacheView,
        slot_mapping: torch.Tensor
):
    """将 K/V 存入 KV Cache。

    默认 fp16 路径继续使用当前 Triton store kernel。
    optional quantized 路径在写入前做量化，并维护 scale buffers。
    """

    if isinstance(kv_cache, KVCacheView):
        if kv_cache.is_quantized:
            store_kv_cache_for_attention(k, v, kv_cache, slot_mapping)
            return
        kv_cache = kv_cache.cache

    num_tokens, num_heads, head_dim = k.shape
    block_size = kv_cache.shape[2]

    k_cache = kv_cache[0]
    v_cache = kv_cache[1]

    k = k.contiguous()
    v = v.contiguous()

    grid = (num_tokens,)
    BLOCK_H = min(32, num_heads)
    BLOCK_D = min(32, head_dim)

    store_kvcache_kernel[grid](
        k, v,
        k_cache, v_cache,
        slot_mapping,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D
    )
```

### 7.3 替换 `_decode_attention()` 里的 cache 读取逻辑

把当前：

```python
kv_cache = context.kv_cache[self.layer_idx]
k_cache = kv_cache[0]
v_cache = kv_cache[1]
```

换成：

```python
kv_cache = context.kv_cache[self.layer_idx]
k_cache, v_cache = get_kv_cache_for_attention(
    kv_cache,
    target_dtype=torch.float16,
)
```

完整 `_decode_attention()` 形态如下：

```python
def _decode_attention(
        self,
        q: torch.Tensor,
        context: Context
) -> torch.Tensor:
    """Decode: 使用 flash_attn_with_kvcache。

    optional quantized KV cache 路径会在这里反量化为 fp16 cache，
    再交给 FlashAttention。FlashAttention 本身仍消费 fp16/bf16 tensor。
    """

    original_dtype = q.dtype
    kv_cache = context.kv_cache[self.layer_idx]
    k_cache, v_cache = get_kv_cache_for_attention(
        kv_cache,
        target_dtype=torch.float16,
    )

    q = q.unsqueeze(1).to(torch.float16)

    cache_seqlens = context.context_lens.to(torch.int32)
    block_table = context.block_tables.to(torch.int32)

    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=self.scale,
        causal=True,
    )

    return output.squeeze(1).to(original_dtype)
```

### 7.4 Prefill 路径的语义

`Attention.forward()` 里仍然先调用 `store_kvcache()`，再根据 `context.is_prefill` 选择 prefill / decode：

```python
if context.kv_cache is not None and context.slot_mapping is not None:
    store_kvcache(
        k, v,
        context.kv_cache[self.layer_idx],
        context.slot_mapping,
    )
```

因此：

- prefill 阶段：prompt 的 K/V 会写入 cache；attention 计算仍直接使用当前 step 的浮点 `k` / `v`。
- decode 阶段：新 token 的 K/V 先写入 cache；随后 decode attention 读取完整 cache，并在量化路径下反量化后喂给 FlashAttention。

---

## 8. 新增 `tests/test_Day12_kvcache_quant.py`

新建 `tests/test_Day12_kvcache_quant.py`，写入下面完整测试文件。测试只使用 CPU 小张量，不依赖真实大模型，不导入 `layers/attention.py`，避免把 FlashAttention / Triton 环境要求混进 Day12 协议测试。

```python
"""Day12 optional KV cache quantization tests.

这些测试覆盖：
1. int8_sim / pseudo_fp8_sim roundtrip
2. per-block/per-head scale 计算
3. 默认 fp16 路径保持不变
4. attention store/dequant hook 的形状语义
"""

import sys

sys.path.insert(0, ".")

import torch

from utils.kvcache_quant import (
    allocate_kvcache_view,
    dequantize_block_head,
    estimate_kvcache_bytes_per_layer,
    get_kv_cache_for_attention,
    get_quant_spec,
    quantize_block_head,
    store_kv_cache_for_attention,
)


def test_roundtrip_for_int8_and_pseudo_fp8_sim():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 8, dtype=torch.float32) * 0.25

    for scheme in ["int8_sim", "pseudo_fp8_sim"]:
        q, scales = quantize_block_head(x, scheme)
        restored = dequantize_block_head(q, scales, target_dtype=torch.float32)

        spec = get_quant_spec(scheme)
        assert q.shape == x.shape
        assert q.dtype == spec.storage_dtype
        assert scales.shape == (2, 1)
        assert torch.all(scales > 0)
        assert torch.allclose(restored, x, atol=0.08, rtol=0.25)


def test_scale_uses_per_block_per_head_amax():
    x = torch.tensor(
        [
            [[1.0, -2.0, 3.0], [0.5, -1.0, 2.0]],
            [[-6.0, 0.0, 1.0], [4.0, -3.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    _, scales = quantize_block_head(x, "int8_sim")
    expected = torch.tensor([[6.0 / 127.0], [4.0 / 127.0]], dtype=torch.float32)

    assert scales.shape == (2, 1)
    assert torch.allclose(scales, expected, rtol=1e-6, atol=1e-6)


def test_default_fp16_path_keeps_cache_dtype_and_no_scales():
    view = allocate_kvcache_view(
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        kv_cache_dtype="fp16",
        kv_cache_quant_scheme="none",
    )

    assert not view.is_quantized
    assert view.cache.dtype == torch.float16
    assert view.cache.shape == (2, 2, 4, 2, 8)
    assert view.scales is None

    k_cache, v_cache = get_kv_cache_for_attention(view, target_dtype=torch.float16)
    assert k_cache.data_ptr() == view.cache[0].data_ptr()
    assert v_cache.data_ptr() == view.cache[1].data_ptr()

    fp16_bytes = estimate_kvcache_bytes_per_layer(
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        kv_cache_dtype="fp16",
        kv_cache_quant_scheme="none",
    )
    assert fp16_bytes == view.nbytes


def test_attention_store_hook_shape_semantics():
    view = allocate_kvcache_view(
        num_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=3,
        device=torch.device("cpu"),
        kv_cache_dtype="quantized",
        kv_cache_quant_scheme="int8_sim",
    )

    k = torch.tensor(
        [
            [[0.10, -0.20, 0.30], [0.05, -0.10, 0.20]],
            [[-0.40, 0.20, 0.00], [0.30, -0.25, 0.10]],
            [[0.15, 0.05, -0.05], [-0.35, 0.20, 0.15]],
        ],
        dtype=torch.float32,
    )
    v = -k
    slot_mapping = torch.tensor([0, 1, 4], dtype=torch.long)

    store_kv_cache_for_attention(k, v, view, slot_mapping)

    assert view.is_quantized
    assert view.cache.dtype == torch.int8
    assert view.cache.shape == (2, 2, 4, 2, 3)
    assert view.scales.shape == (2, 2, 2, 1)

    k_cache, v_cache = get_kv_cache_for_attention(view, target_dtype=torch.float16)
    assert k_cache.shape == (2, 4, 2, 3)
    assert v_cache.shape == (2, 4, 2, 3)
    assert k_cache.dtype == torch.float16
    assert v_cache.dtype == torch.float16

    assert torch.allclose(k_cache[0, 0].float(), k[0], atol=0.02, rtol=0.2)
    assert torch.allclose(k_cache[0, 1].float(), k[1], atol=0.02, rtol=0.2)
    assert torch.allclose(k_cache[1, 0].float(), k[2], atol=0.02, rtol=0.2)
    assert torch.allclose(v_cache[0, 0].float(), v[0], atol=0.02, rtol=0.2)
    assert torch.allclose(v_cache[0, 1].float(), v[1], atol=0.02, rtol=0.2)
    assert torch.allclose(v_cache[1, 0].float(), v[2], atol=0.02, rtol=0.2)


def run_all_tests():
    test_roundtrip_for_int8_and_pseudo_fp8_sim()
    test_scale_uses_per_block_per_head_amax()
    test_default_fp16_path_keeps_cache_dtype_and_no_scales()
    test_attention_store_hook_shape_semantics()
    print("Day12 KV cache quantization tests passed.")


if __name__ == "__main__":
    run_all_tests()
```

---

## 9. 验收命令

在 `nano_vll_repro/` 目录运行：

先跑最小协议命令：

```bash
python -m py_compile utils/kvcache_quant.py tests/test_Day12_kvcache_quant.py
python tests/test_Day12_kvcache_quant.py
```

再跑覆盖所有修改文件的全量编译：

```bash
python -m py_compile config.py utils/kvcache_quant.py engine/model_runner.py utils/context.py layers/attention.py tests/test_Day12_kvcache_quant.py
```

最小协议命令不要求本地有 Qwen 模型目录，也不加载真实大模型；全量编译只检查主线接入文件的语法。

---

## 10. 常见坑

1. **把 KV cache 量化和 weight quantization 混成一个配置。**
   Day12 的配置只管理 KV cache storage，权重量化不在这条路径里。
2. **认为换 dtype 就等于支持 FP8。**
   KV cache 量化需要 container、scale、写入协议、读取协议一起定义。
3. **让 FlashAttention 直接读取教学版低精度 cache。**
   这里的 FlashAttention decode hook 会先反量化到 `fp16`；这不是生产级 FP8 cache kernel。
4. **默认开启量化路径。**
   教学仓库要保持默认行为稳定，量化路径必须由配置显式开启。
5. **只测试 roundtrip，不测试默认路径。**
   optional path 的第一条验收标准是默认 `fp16` 路径不变。

---

## 11. 读完你应该明白

1. `kv_cache_dtype="fp16"` + `kv_cache_quant_scheme="none"` 为什么保持当前主线行为。
2. `int8_sim` / `pseudo_fp8_sim` 如何通过低精度 container + per-block/per-head scale 表达 KV cache 量化。
3. `ModelRunner.allocate_kv_cache()` 为什么需要把量化 cache 和 scale buffers 按层绑定。
4. `Context.kv_cache` 为什么需要表达 `torch.Tensor | KVCacheView`。
5. `layers/attention.py` 为什么是“写前量化、读前反量化”，而不是直接声明支持生产级 FP8 FlashAttention。

下一篇：`Day13-CPU-KV-Block-Offload.md`（内容是 CPU KV block swap）。
