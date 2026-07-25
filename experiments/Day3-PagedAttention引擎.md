# Day 3 — PagedAttention 引擎（核心灵魂）

## 本篇定位

这是整个项目**最重要**的一篇。PagedAttention 是 vLLM 论文的核心贡献，也是 nano-vLLM 区别于普通 Transformer 推理代码的关键。

如果你只能认真读一篇，就读这篇。理解了 Day3，其他都是细节。

---

## 1. 📖 知识点讲解

### 1.1 传统 KV Cache 管理的问题

先回忆一下 KV Cache 是干什么的：

```
生成 "我 爱 北京 天安门" 时：

Step 1: 输入 "我"       → 算 K₁V₁，存入 cache
Step 2: 输入 "爱"       → 只算 K₂V₂，从 cache 读 K₁V₁，做 attention
Step 3: 输入 "北京"     → 只算 K₃V₃，从 cache 读 K₁V₁+K₂V₂，做 attention
Step 4: 输入 "天安门"   → 只算 K₄V₄，从 cache 读 K₁V₁+K₂V₂+K₃V₃，做 attention
```

**传统做法：为每个请求预留一整块连续显存。**

```
请求A（prompt 100 tokens, 最多生成 500 tokens）:
[████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
 └── 预留 600 token 的连续空间 ──────────────────────────────────────┘

请求B（prompt 800 tokens, 最多生成 100 tokens）:
[████████████████████████████████████████████████████████████████████████████████░░░░░░░]
 └── 预留 900 token 的连续空间 ──────────────────────────────────────────────────────┘
```

**问题**：
1. **内部碎片**：请求A 只用了 100 个位置，剩下 500 个全浪费
2. **外部碎片**：请求A 释放后留出一个空洞，请求C 可能因为空洞不够大而分配失败
3. **无法共享**：请求A 和请求B 的 prompt 前缀相同（比如相同的 system prompt），但它们的 KV Cache 是各自独立的，重复计算+重复存储

### 1.2 PagedAttention 的解决方案

**核心思想：把 KV Cache 切成固定大小的「页」(block)，像操作系统的虚拟内存一样管理。**

```
传统: 每个请求独占一块连续内存
┌─────────────────┐  ┌─────────────────┐
│   请求A 的 KV    │  │   请求B 的 KV    │
│   (连续一大块)   │  │   (连续一大块)   │
└─────────────────┘  └─────────────────┘

PagedAttention: 内存被切成固定大小的页，按需分配，可以不连续
物理页池:  [页0] [页1] [页2] [页3] [页4] [页5] [页6] [页7] ...
              │     │           │     │
              ▼     ▼           ▼     ▼
请求A:      [页0] [页2]       请求B: [页1] [页5]
(逻辑块0→物理页0)              (逻辑块0→物理页1)
(逻辑块1→物理页2)              (逻辑块1→物理页5)
```

**好处**：
- **零内部碎片**：最后一个块没填满最多浪费 block_size-1 个 token 的空间（256 个 token ≈ 可忽略）
- **零外部碎片**：物理页可以任意分配，不需要连续空间
- **前缀共享**：两个请求的 prompt 相同部分可以指向同一个物理页，只需引用计数 +1

### 1.3 五个关键数据结构（复习）

| 概念 | 在代码中的位置 | 作用 |
|------|-------------|------|
| **Block** | `engine/block_manager.py` | 一个物理页。包含 `block_id`、`ref_count`（引用计数）、`hash`（内容哈希）、`token_ids` |
| **BlockManager** | `engine/block_manager.py` | 管理所有物理页的分配/释放，维护空闲池、已用集合、Prefix Cache 哈希表 |
| **block_table** | `Sequence.block_table` | 逻辑块→物理页的映射表。`[17, 203, 41]` 表示该序列的 3 个逻辑块分别映射到物理页 17、203、41 |
| **slot_mapping** | 由 `BlockManager.get_slot_mapping()` 计算 | 每个 token 在 KV Cache 中的精确位置。`slot = 物理页号 × block_size + 页内偏移` |
| **Context** | `utils/context.py` | 全局单例，把本轮的所有元数据打包传给 Attention 层 |

### 1.4 block_table 与 slot_mapping 的区别（最容易混淆）

```python
# block_table: 以「页」为单位的粗粒度映射
seq.block_table = [17, 203, 41]
# 意思是：逻辑块0→物理页17，逻辑块1→物理页203，逻辑块2→物理页41

# slot_mapping: 以「token」为单位的细粒度映射
# 从 block_table 推导：
#   物理页17: slot范围 [17*256, 17*256+255] = [4352, 4353, ..., 4607]
#   物理页203: slot范围 [203*256, 203*256+255] = [51968, ..., 52223]
slot_mapping = [4352, 4353, ..., 4607, 51968, ...]
```

**记忆口诀**：`block_table` 给调度器用（粗粒度，一页一页管理），`slot_mapping` 给 Attention 写 KV Cache 用（细粒度，一个 token 一个槽）。

### 1.5 Prefix Cache（前缀缓存）

两条请求共享相同的系统提示：

```
请求A: "You are a helpful assistant. 解释量子力学"
请求B: "You are a helpful assistant. 写一首诗"
                              ↑ 前 N 个 token 完全一样
```

Prefix Cache 机制：

1. 每个完整的物理块在填满时计算一个**内容哈希**（xxhash）
2. 哈希存入 `hash_to_block_id` 字典
3. 新请求分配块时，先检查是否有内容完全相同的已存在块
4. 如果有 → **引用计数 +1**，不分配新块（节省显存 + 跳过计算）
5. 如果没有 → 分配新块，填满后记录哈希

```python
# 简化的 Prefix Cache 流程
def allocate(seq):
    for each block:
        token_hash = compute_hash(token_ids)
        if token_hash in hash_to_block_id:
            # 命中！复用已有物理页
            cached_block = blocks[hash_to_block_id[token_hash]]
            cached_block.ref_count += 1
            seq.block_table.append(cached_block.block_id)
            seq.num_cached_tokens += block_size  # 标记这些 token 不需要重新算
        else:
            # 未命中，分配新页
            new_block = allocate_fresh_block()
            seq.block_table.append(new_block.block_id)
            hash_to_block_id[token_hash] = new_block.block_id
```

### 1.6 Prefill 和 Decode 中 Attention 的差异

| | Prefill（预填充） | Decode（逐 token 生成） |
|---|---|---|
| **输入** | 整段 prompt（可能上百个 token） | 上一个 step 生成的 1 个 token |
| **Q 的长度** | prompt_len 个 token | 1 个 token |
| **KV Cache 操作** | **写入**：整段 prompt 的 K/V 写入 cache | **追加写入**：1 个新 token 的 K/V |
| **FlashAttention API** | `flash_attn_varlen_func(q, k, v, ...)` | `flash_attn_with_kvcache(q, k_cache, v_cache, ...)` |
| **计算复杂度** | O(prompt_len²) | O(prompt_len) — 因为 KV 都从 cache 读 |

关键区别在于 FlashAttention 的 API：
- **Prefill**：Q、K、V 都是新算出来的张量，直接传给 `flash_attn_varlen_func`。这个 API 支持变长序列（不同请求的 prompt 长度不同），通过 `cu_seqlens` 标记边界。
- **Decode**：Q 是新算的（1个token），K 和 V 从 KV Cache 中读取，用 `flash_attn_with_kvcache`。这个 API 需要 `block_tables` 来定位物理页。

---

## 2. 🔍 已有代码回顾

### 2.1 Block（`engine/block_manager.py`）— 已有实现

```python
class Block:
    """物理内存块"""
    def __init__(self, block_id: int):
        self.block_id = block_id      # 物理页编号
        self.ref_count = 0            # 引用计数（多少序列在用这个块）
        self.hash = -1                # 内容哈希（-1 表示未完成/无效）
        self.token_ids = []           # 存储的 token ID（用于 Prefix Cache 验证）

    def update(self, hash_value, token_ids):
        """块填满时更新哈希和内容"""
        self.hash = hash_value
        self.token_ids = token_ids

    def reset(self):
        """重新分配时重置"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

**设计要点**：
- `ref_count`：支持多个序列共享同一个物理块（Prefix Cache 的核心）。只有引用计数归零时才真正释放。
- `hash = -1`：用 -1 表示「未被完整填充的块」——没有有效哈希的块不能用于 Prefix Cache。
- `token_ids`：存储原始 token ID 列表，用于**验证**哈希匹配后确实内容相同（防止哈希碰撞）。

### 2.2 BlockManager（`engine/block_manager.py`）— 已有实现

BlockManager 有三个关键的内部方法，它们的区别需要搞清楚：

```python
# 方法 1：_allocate_fresh_block() — 从空闲池头部取新块，彻底清空

# 方法 2：_recover_block(block_id) — 复活空闲池中的一个指定块，保留其哈希和内容
#   （用于 Prefix Cache 命中：这个块之前被某序列用过，现在空闲了，但内容还在）

# 方法 3：_deallocate_block(block_id) — 释放块回空闲池
#   （引用计数归零时调用）
```

三个方法的协作关系：
```
新序列请求分配 → allocate(seq)
  ├─ block 0: compute_hash → 命中 Prefix Cache？
  │   ├─ YES → _recover_block(block_id)  // 复活空闲块，保留内容
  │   └─ NO  → _allocate_fresh_block()    // 取新块，清空
  ├─ block 1: 同上...
  └─ ...

序列完成 → deallocate(seq)
  └─ 引用计数 -1 → 归零则 _deallocate_block(block_id)
```

### 2.3 store_kvcache Triton Kernel（`layers/attention.py`）— 已有实现

```python
@triton.jit
def store_kvcache_kernel(
    K, V, KCache, VCache, slot_mapping,
    stride_kn, stride_kh, stride_kd,     # K 张量的 stride
    stride_vn, stride_vh, stride_vd,     # V 张量的 stride
    stride_kcb, stride_kcs, stride_kch, stride_kcd,  # KCache 的 stride
    stride_vcb, stride_vcs, stride_vch, stride_vcd,  # VCache 的 stride
    num_heads, head_dim, block_size,
    BLOCK_H, BLOCK_D                     # Triton 分块计算参数
):
    """每个 program 处理一个 token，将其 K/V 写入 Cache 的指定 slot"""
    token_idx = tl.program_id(0)         # 我是第几个 token
    slot = tl.load(slot_mapping + token_idx)  # 从 slot_mapping 读取目标位置

    block_id = slot // block_size        # 目标物理页编号
    block_offset = slot % block_size     # 页内偏移

    # 遍历所有 head 和 head_dim，分块加载/存储
    for h in range(0, num_heads, BLOCK_H):
        for d in range(0, head_dim, BLOCK_D):
            k = tl.load(K + ...)         # 从输入 K 读取
            v = tl.load(V + ...)         # 从输入 V 读取
            tl.store(KCache + ..., k)    # 写入 KV Cache 的 K 部分
            tl.store(VCache + ..., v)    # 写入 KV Cache 的 V 部分
```

**为什么用 Triton kernel 而不是 PyTorch 原生索引？**

PyTorch 的 `kv_cache[slot_mapping] = k` 看似简单，但涉及 GPU 上的 scatter 操作（非连续写入），性能不如自定义 Triton kernel。Triton 可以精确控制内存访问模式。

### 2.4 Attention 类（`layers/attention.py`）— 已有实现

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        # 步骤1: 无论 prefill 还是 decode，先存入 KV Cache
        if context.slot_mapping is not None:
            store_kvcache(k, v, context.kv_cache[self.layer_idx], context.slot_mapping)

        # 步骤2: 根据阶段选择不同的 FlashAttention API
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            return self._decode_attention(q, context)
            # 注意：decode 时不传 k, v，因为 K/V 已经从 cache 读取

    def _prefill_attention(self, q, k, v, context):
        return flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )

    def _decode_attention(self, q, context):
        k_cache = context.kv_cache[self.layer_idx][0]  # K cache
        v_cache = context.kv_cache[self.layer_idx][1]  # V cache
        return flash_attn_with_kvcache(
            q=q.unsqueeze(1),  # [num_seqs, num_heads, head_dim] → [num_seqs, 1, num_heads, head_dim]
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        ).squeeze(1)
```

---

## 3. ⚠️ 当前代码的问题分析

BlockManager 和 Attention 的代码已经比较完整。主要的问题在于：

1. **`store_kvcache` Python wrapper 没有定义 stride 参数**：当前代码直接传了 `k.stride(0)` 等，但如果 K 不是连续内存，stride 可能出错。现有代码已经做了 `.contiguous()`，所以暂时没问题。

2. **Attention 层把 QKV 传给 `flash_attn_varlen_func` 时强制转 `float16`**：如果模型是 bf16，这里会有一次类型转换开销。当前代码做了 `.to(torch.float16)` 再 `.to(q.dtype)` 转回，这是因为某些旧版 flash-attn 不完全支持 bf16。

3. **BlockManager 的 `_recover_block` 使用的是 `list.remove()`**：时间复杂度 O(n)，在大规模场景下可能有性能影响。但对于教学项目足够。

4. **🔴 `block_manager.py` 第 298 行 Off-by-One 错误**：Prefix Cache 链式哈希的条件判断有误。

```python
# ❌ 当前代码（第 298 行）
prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) > 2 else -1

# ✅ 正确应为
prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) >= 2 else -1
```

**影响**：当 `block_table` 恰好有 2 个块时（如 `[0, 1]`），第一个块填满后计算哈希时，`len(block_table) > 2` 为 `False`，导致 `prefix_hash = -1`。但正确的逻辑应该是使用 block 0 的哈希作为 block 1 的前缀。这会破坏 Prefix Cache 的链式哈希，导致缓存命中率下降。

**验证方法**：
```python
# 假设 block_table = [0, 1]，第一个块刚填满
# 当前代码: prefix_hash = -1 （错误，丢失了 block 0 的哈希链）
# 正确代码: prefix_hash = blocks[0].hash （正确，保持哈希链）
```

> 注意：以上问题 1-3 不影响正确性，问题 4 影响 Prefix Cache 的正确性。下面的完整代码已包含问题 4 的修复（`>= 2`）。

---

## 4. 📝 完整代码

以下代码就是你当前 `engine/block_manager.py` 和 `layers/attention.py` 的完整内容。三个月没看，建议逐段阅读注释。

### 4.1 `engine/block_manager.py`

```python
"""Block Manager - PagedAttention 核心组件

实现类似操作系统分页的 KV Cache 管理：
- Block：物理内存块，固定大小（默认 256 tokens）
- BlockManager：管理 block 的分配、释放、复用

关键概念：
- 物理块(Physical Block)：GPU 显存中实际存储 KV 的位置
- 逻辑块(Logical Block)：序列视角看到的块索引
- block_table：逻辑块 → 物理块的映射表（存在 Sequence 里）
- slot_mapping：token 位置 → 物理 cache 槽位的映射（由 BlockManager 计算）

类比操作系统：
- 物理块 = 物理内存页框（page frame）
- 逻辑块 = 虚拟内存页（virtual page）
- block_table = 页表（page table）
- slot_mapping = 物理地址
- BlockManager = 物理内存管理器
- Prefix Cache = 共享内存映射（mmap）
"""

from collections import deque
from engine.sequence import Sequence

try:
    import xxhash
    import numpy as np
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    print("[警告] xxhash not installed, using builtin hash (slower)")


class Block:
    """物理内存块
    
    每个 Block 可存储 block_size 个 token 的 KV Cache。
    类比操作系统的「物理页框」(page frame)。
    
    Attributes:
        block_id: 物理块 ID（在 kv_cache tensor 中的索引）
        ref_count: 引用计数 — 多少序列在使用这个块（支持 Prefix Cache 共享）
        hash: 内容哈希 — 用于 Prefix Cache 快速查找（-1 表示无效/未完成）
        token_ids: 存储的 token ID 列表 — 用于验证 cache 命中时内容确实相同
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0          # 初始未被使用
        self.hash = -1              # -1 = "这个块还没被完整填充，不能用于 Prefix Cache"
        self.token_ids = []         # 只在块填满时记录（用于哈希碰撞后的内容验证）

    def update(self, hash_value: int, token_ids: list[int]):
        """块被完整填充时调用：记录哈希和内容"""
        self.hash = hash_value
        self.token_ids = token_ids

    def reset(self):
        """重新分配时调用：清空旧数据，重置引用计数为 1"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def __repr__(self):
        return f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash})"


class BlockManager:
    """Block 管理器（整个 PagedAttention 系统的核心）
    
    核心职责：
    1. 维护空闲/已用 block 池
    2. 为序列分配/释放 blocks（allocate / deallocate）
    3. 支持 Prefix Caching（基于内容哈希复用 block）
    4. 支持 decode 阶段的逐 token 追加（append_slot）
    5. 计算 slot_mapping（token → cache 槽位）
    
    数据结构：
    - blocks: 所有物理 block 的列表（不变）
    - free_block_ids: 空闲 block ID 的双端队列（O(1) 取头部）
    - used_block_ids: 已用 block ID 的集合（O(1) 查找）
    - hash_to_block_id: 内容哈希 → block ID 的字典（Prefix Cache 核心）
    
    三个内部分配方法的区别（重要！）:
    1. _allocate_fresh_block(): 从空闲池头部取一个新块，调用 reset() 彻底清空
       → 用于 Cache Miss：需要全新的块
    2. _recover_block(block_id): 复活空闲池中指定 ID 的块，保留原有 hash 和 token_ids
       → 用于 Cache Hit：块之前在空闲池里但内容还有效，直接复用
    3. _deallocate_block(block_id): 释放块回空闲池
       → 用于引用计数归零时
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 总 block 数（根据 GPU 显存计算）
            block_size: 每个 block 存储的 token 数（默认 256）
        """
        self.block_size = block_size
        self.num_blocks = num_blocks

        # 创建所有物理块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 空闲队列：deque 支持 O(1) 从两端取/放
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 已用集合：set 支持 O(1) 检查是否在使用中
        self.used_block_ids: set[int] = set()

        # Prefix Cache 的核心：哈希 → 物理块 ID
        # 只有当块被完整填充（有 block_size 个 token）时才会进入这个表
        self.hash_to_block_id: dict[int, int] = {}

    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """计算 token 序列的内容哈希值（用于 Prefix Cache 匹配）
        
        支持链式哈希（rolling hash）：当前块的哈希依赖于前一个块的哈希，
        确保「相同前缀」的不同序列能够匹配到相同的物理块。
        
        为什么要链式？
          请求A: [system_prompt | user_msg_1]
          请求B: [system_prompt | user_msg_2]
        
        如果不链式，system_prompt 的块哈希只取决于自身内容 → 可以匹配 ✓
        如果链式，则第二块的哈希 = hash(prefix_hash + 自身内容)，其中 prefix_hash 相同
        → 只要前缀相同，后续块的哈希也自动相同 → 更稳健 ✓
        
        Args:
            token_ids: 当前块的 token ID 列表
            prefix_hash: 前一个块的哈希值（-1 表示这是第一个块，无前缀）
        
        Returns:
            64-bit 哈希值（xxhash 或 Python 内置 hash）
        """
        if HAS_XXHASH:
            h = xxhash.xxh64()
            if prefix_hash != -1:
                # 把前一块的哈希混入当前块的计算中 → 链式依赖
                h.update(prefix_hash.to_bytes(8, "little"))
            # 把当前块的 token ID 作为字节写入
            h.update(np.array(token_ids, dtype=np.int64).tobytes())
            return h.intdigest()
        else:
            # 降级方案：Python 内置 hash
            return hash((prefix_hash, tuple(token_ids)))

    # ═══════════════════════════════════════════════════════════════
    # 内部方法：块的分配与释放
    # ═══════════════════════════════════════════════════════════════

    def _allocate_fresh_block(self) -> Block:
        """从空闲池头部取一个新块，并彻底清空旧数据。
        
        时间复杂度：O(1) — 使用 deque.popleft()
        
        使用场景：Prefix Cache 未命中，需要一个全新的块来存储新内容。
        
        关键动作：
        1. 从 free_block_ids 头部取 block_id
        2. 调用 block.reset() → 清空 hash 和 token_ids，设置 ref_count=1
        3. 将 block_id 移入 used_block_ids
        """
        if not self.free_block_ids:
            raise RuntimeError("显存耗尽！没有可用的空闲块了。")

        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} 仍在被使用中，不能分配"

        block.reset()  # 彻底清空
        self.used_block_ids.add(block_id)
        block.ref_count = 1

        return block

    def _recover_block(self, block_id: int) -> Block:
        """复活空闲池中指定 ID 的块，保留其原有内容（hash 和 token_ids）。
        
        时间复杂度：O(n) — list.remove() 是线性的
        
        使用场景：Prefix Cache 命中 — 该块之前被用过且已归还空闲池，
        但内容（hash + token_ids）仍然有效，不需要重新计算。
        
        与 _allocate_fresh_block 的区别：
        ❌ 不调用 reset() → 保留 hash 和 token_ids
        ✅ 仅恢复 ref_count=1，移入 used_block_ids
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} 仍在被使用中，不能恢复"

        self.free_block_ids.remove(block_id)  # O(n)，教学版可接受
        self.used_block_ids.add(block_id)
        block.ref_count = 1
        return block

    def _deallocate_block(self, block_id: int):
        """释放一个块回空闲池（仅在引用计数归零时由 deallocate 调用）"""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} 还有引用，不能释放"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        # 注意：这里不重置 hash！块虽然归还了空闲池，但内容还在。
        # 这使得 Prefix Cache 可以在后续请求中「复活」这个块。

    # ═══════════════════════════════════════════════════════════════
    # 公开方法：序列级别的操作
    # ═══════════════════════════════════════════════════════════════

    def get_num_free_blocks(self) -> int:
        """获取当前空闲 block 数量"""
        return len(self.free_block_ids)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲 blocks 来容纳这个序列的所有 token。
        
        这是一个「悲观检查」——只看空闲块数够不够，不考虑 Prefix Cache 命中。
        真实的 vLLM 会估算命中率来减少预留。
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为序列分配所有需要的物理块（Prefill 阶段调用）。
        
        这是 Prefix Cache 的核心实现！流程：
        
        对序列的每个逻辑块（按 block_size 切分）：
          1. 计算该块 token 的内容哈希
          2. 检查 hash_to_block_id 是否有匹配的已存在块
          3. 如果命中（缓存验证通过）→ 复用该块，ref_count += 1
          4. 如果未命中 → 分配新块，如果是完整块则记录到 hash_to_block_id
        
        重要设计决策：「一旦缓存未命中，后续所有块都不再尝试匹配」
        （因为链式哈希使得未命中后的哈希值不可预测）
        
        Args:
            seq: 要分配的序列（其 block_table 将被填充）
        """
        assert not seq.block_table, "序列已有 block_table，不能重复分配"

        prefix_hash = -1          # 链式哈希的前缀值
        cache_miss = False        # 一旦未命中，后续块也不再尝试 Prefix Cache

        for i in range(seq.num_blocks):
            # 获取第 i 个逻辑块的 token 列表
            token_ids = seq.block(i)  # Sequence.block(i) 返回 token_ids[i*block_size : (i+1)*block_size]

            # 只有完整块（token 数 == block_size）才参与 Prefix Cache
            is_full_block = (len(token_ids) == self.block_size)

            # 计算哈希（仅在完整块且之前没有未命中的情况下）
            current_hash = (
                self.compute_hash(token_ids, prefix_hash)
                if is_full_block and not cache_miss
                else -1
            )

            # 查询 Prefix Cache
            cached_block_id = self.hash_to_block_id.get(current_hash, -1)

            # ── 分支 A：尝试缓存命中 ──
            if cached_block_id != -1:
                cached_block = self.blocks[cached_block_id]
                # 二次验证：哈希可能冲突，必须比对实际 token ID
                if cached_block.token_ids == token_ids:
                    # ✅ 缓存命中！
                    seq.num_cached_tokens += self.block_size  # 标记这些 token 不需要重新计算

                    if cached_block_id in self.used_block_ids:
                        # 块正在被其他序列使用 → 增加引用计数
                        cached_block.ref_count += 1
                    else:
                        # 块在空闲池中 → 复活它（保留内容）
                        self._recover_block(cached_block_id)

                    seq.block_table.append(cached_block_id)
                    prefix_hash = current_hash      # 更新链式哈希前缀
                    continue  # 跳过未命中分支，继续下一个逻辑块

            # ── 分支 B：未命中（或哈希冲突导致的假命中）──
            cache_miss = True  # 一旦未命中，后续块不再尝试（保持链式哈希一致性）

            block = self._allocate_fresh_block()
            block_id = block.block_id

            # 如果是完整块，记录到 Prefix Cache 供后续请求复用
            if is_full_block:
                if current_hash == -1:
                    # 之前未命中导致没算哈希，现在补算
                    current_hash = self.compute_hash(token_ids, prefix_hash)
                block.update(current_hash, token_ids.copy())
                self.hash_to_block_id[current_hash] = block_id
                prefix_hash = current_hash

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有物理块。
        
        引用计数 -1，只有当计数归零时才真正释放回空闲池。
        这种设计使得 Prefix Cache 共享的块不会因为一个序列完成而被错误释放。
        
        Args:
            seq: 要释放的序列
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1

            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以追加一个 token 的 slot（Decode 阶段调用）。
        
        大多数时候不需要新块（因为 token 还落在当前块内），
        只有当序列长度刚好超出当前块的边界时才需要分配新块。
        
        条件：len(seq) % block_size == 1
        含义：上一个 token 刚好填满了一个块，新 token 需要新的块。
        """
        needs_new_block = (len(seq) % self.block_size == 1)
        return len(self.free_block_ids) >= needs_new_block

    def append_slot(self, seq: Sequence):
        """为新生成的 token 分配 KV Cache 槽位（Decode 阶段调用）。
        
        三种情况：
        1. 刚好需要新 block（position_in_block == 1）
           → 分配一个全新的块，追加到 block_table
        
        2. 刚好填满了当前 block（position_in_block == 0）
           → 当前块现在完整了，计算哈希并加入 Prefix Cache
        
        3. 还在当前 block 内（position_in_block 在 2~block_size-1）
           → 无需任何操作，槽位在已分配的块内自动可用
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        position_in_block = len(seq) % self.block_size

        if position_in_block == 1:
            # 情况1：需要新块。上一个块必须是完整的（有有效哈希）
            assert last_block.hash != -1, "上一个块应该已经完整并记录了哈希"
            new_block = self._allocate_fresh_block()
            block_table.append(new_block.block_id)

        elif position_in_block == 0:
            # 情况2：当前块刚好填满。计算链式哈希并加入 Prefix Cache
            assert last_block.hash == -1, "这个块应该还没有哈希"
            token_ids = seq.block(seq.num_blocks - 1)  # 最后一个块的 token

            # 链式哈希：取倒数第二个块的哈希作为前缀
            prefix_hash = (
                self.blocks[block_table[-2]].hash
                if len(block_table) >= 2
                else -1
            )
            current_hash = self.compute_hash(token_ids, prefix_hash)

            last_block.update(current_hash, token_ids.copy())
            self.hash_to_block_id[current_hash] = last_block.block_id

        # 情况3：什么都不做，slot 在当前块内

    def get_slot_mapping(self, seq: Sequence, start_pos: int = 0) -> list[int]:
        """计算从 start_pos 开始的所有 token 在 KV Cache 中的精确槽位。
        
        slot = block_id × block_size + offset_in_block
        
        这个函数是整个系统正确性的关键：如果 slot 算错，
        KV Cache 写入位置全乱，输出直接坏掉。
        
        Args:
            seq: 目标序列
            start_pos: 起始位置（Preill 时为 0，Decode 时为 num_tokens-1）
        
        Returns:
            slot 列表，长度 = len(seq) - start_pos
        """
        slots = []
        for pos in range(start_pos, len(seq)):
            block_idx = pos // self.block_size           # 第几个逻辑块
            offset = pos % self.block_size               # 块内偏移
            block_id = seq.block_table[block_idx]        # 查表：逻辑块→物理页
            slot = block_id * self.block_size + offset   # 全局槽位
            slots.append(slot)
        return slots

    def __repr__(self):
        return (
            f"BlockManager(num_blocks={self.num_blocks}, "
            f"free={len(self.free_block_ids)}, "
            f"used={len(self.used_block_ids)})"
        )
```

### 4.2 `layers/attention.py`

```python
"""PagedAttention 层

整合两个关键能力：
1. Triton store_kvcache kernel：将 K/V 写入 KV Cache 的指定 slot
2. FlashAttention：高效计算 attention（Prefill 和 Decode 使用不同的 API）

两种模式：
- Prefill: flash_attn_varlen_func — 支持变长序列的并行 attention
- Decode: flash_attn_with_kvcache — 从 KV Cache 读取历史，只算新 token

全局 Context 的作用：
Attention 层不直接接收 slot_mapping、block_tables 等参数，
而是通过 get_context() 从全局单例中读取。
这避免了修改所有中间层（DecoderLayer、Qwen3Model）的 forward 签名。
"""

import torch
from torch import nn
from utils.context import get_context, Context

import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


# ══════════════════════════════════════════════════════════════════
# Triton Kernel：将 K/V 写入 KV Cache
# ══════════════════════════════════════════════════════════════════

@triton.jit
def store_kvcache_kernel(
    # 输入张量
    K, V,
    # 目标 Cache 张量
    KCache, VCache,
    # 每个 token 的目标槽位
    slot_mapping,
    # K 张量的 stride（用于计算内存地址）
    stride_kn, stride_kh, stride_kd,
    # V 张量的 stride
    stride_vn, stride_vh, stride_vd,
    # KCache 的 stride
    stride_kcb, stride_kcs, stride_kch, stride_kcd,
    # VCache 的 stride
    stride_vcb, stride_vcs, stride_vch, stride_vcd,
    # 常量参数（Triton 要求用 tl.constexpr 标注编译时常量）
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr,   # 每次处理的 head 数（分块大小）
    BLOCK_D: tl.constexpr,   # 每次处理的 head_dim 数（分块大小）
):
    """Triton kernel：将一批 token 的 K/V 写入 KV Cache 的指定 slot。
    
    每个 CUDA thread block 处理一个 token。
    在 thread block 内部，用 2D grid 分块处理 head 和 head_dim 维度。
    
    为什么会需要这个 kernel？
    PyTorch 的 kv_cache[slot_mapping] = k 看似简单，但 scatter 写入
    （非连续地址写入）性能不好。Triton 可以精确控制访问模式。
    """
    # 我是第几个 token
    token_idx = tl.program_id(0)

    # 从 slot_mapping 读取这个 token 的目标槽位
    slot = tl.load(slot_mapping + token_idx)

    # 计算目标物理页和页内偏移
    block_id = slot // block_size
    block_offset = slot % block_size

    # 分块遍历所有 head 和 head_dim
    for h in range(0, num_heads, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < num_heads

        for d in range(0, head_dim, BLOCK_D):
            d_offsets = d + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < head_dim

            # 组合 head 和 dim 的 mask
            mask = h_mask[:, None] & d_mask[None, :]

            # ── 从输入读取 K ──
            k_ptrs = (
                K
                + token_idx * stride_kn
                + h_offsets[:, None] * stride_kh
                + d_offsets[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask, other=0.0)

            # ── 从输入读取 V ──
            v_ptrs = (
                V
                + token_idx * stride_vn
                + h_offsets[:, None] * stride_vh
                + d_offsets[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask, other=0.0)

            # ── 写入 KV Cache 的 K 部分 ──
            kc_ptrs = (
                KCache
                + block_id * stride_kcb
                + block_offset * stride_kcs
                + h_offsets[:, None] * stride_kch
                + d_offsets[None, :] * stride_kcd
            )
            tl.store(kc_ptrs, k, mask=mask)

            # ── 写入 KV Cache 的 V 部分 ──
            vc_ptrs = (
                VCache
                + block_id * stride_vcb
                + block_offset * stride_vcs
                + h_offsets[:, None] * stride_vch
                + d_offsets[None, :] * stride_vcd
            )
            tl.store(vc_ptrs, v, mask=mask)


def store_kvcache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Python wrapper：调用 Triton kernel 将 K/V 存入 KV Cache。
    
    Args:
        k: [num_tokens, num_kv_heads, head_dim] — 当前 batch 的 Key
        v: [num_tokens, num_kv_heads, head_dim] — 当前 batch 的 Value
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
                  — 完整 KV Cache（dim=0 时 0=K, 1=V）
        slot_mapping: [num_tokens] — 每个 token 的目标槽位
    """
    num_tokens, num_heads, head_dim = k.shape
    block_size = kv_cache.shape[2]

    # 分离 K Cache 和 V Cache
    k_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache = kv_cache[1]

    # 确保内存连续（Triton 需要连续内存来计算 stride）
    k = k.contiguous()
    v = v.contiguous()

    # 启动 Triton kernel：grid = (num_tokens,)，每个 program 处理一个 token
    grid = (num_tokens,)

    # 分块大小：不能超过实际的 head 数和 head_dim
    BLOCK_H = min(32, num_heads)
    BLOCK_D = min(32, head_dim)

    store_kvcache_kernel[grid](
        k, v,
        k_cache, v_cache,
        slot_mapping,
        # K 张量的 stride 信息
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        # KCache 张量的 stride 信息
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # 编译时常量
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )


# ══════════════════════════════════════════════════════════════════
# Attention 层：整合 KV Cache 写入 + FlashAttention
# ══════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """PagedAttention with FlashAttention

    两种模式自动切换：
    - Prefill：使用 flash_attn_varlen_func（处理变长 prompt）
    - Decode：使用 flash_attn_with_kvcache（从 cache 读取历史 KV）

    注意：这些 API 内部都包含了 causal mask，不需要手动创建。
    """

    def __init__(
        self,
        num_heads: int,           # Q 的头数
        head_dim: int,            # 每个头的维度
        scale: float,             # softmax 缩放因子（通常是 1/sqrt(head_dim)）
        num_kv_heads: int,        # KV 的头数（GQA 时 < num_heads）
        layer_idx: int = 0,       # 当前是第几层（用来索引 KV Cache 列表）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx

    def forward(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    ) -> torch.Tensor:
        """Attention 前向传播。

        执行顺序（非常重要）：
        1. 先把 K/V 存入 KV Cache（无论 prefill 还是 decode 都要存）
        2. 然后根据阶段选择不同的 FlashAttention API

        为什么先存后算？因为 flash_attn_with_kvcache 需要读到刚写入的 K/V。
        """
        context = get_context()

        # ══ 步骤 1：将当前 K/V 写入 KV Cache ══
        if context.kv_cache is not None and context.slot_mapping is not None:
            store_kvcache(
                k, v,
                context.kv_cache[self.layer_idx],  # 当前层的 cache
                context.slot_mapping,              # 每个 token 的目标槽位
            )

        # ══ 步骤 2：根据阶段选择 Attention API ══
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            # Decode 阶段不需要传 k, v（它们刚从 cache 中读取）
            return self._decode_attention(q, context)

    def _prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Prefill 阶段的 Attention：使用 flash_attn_varlen_func。

        支持变长序列：不同请求的 prompt 长度可以不同。
        通过 cu_seqlens（累积序列长度）标记每个序列的边界。

        例如：3 条请求，prompt 长度分别是 5, 3, 4
        cu_seqlens = [0, 5, 8, 12]  ← 累积边界
        """
        # flash_attn_varlen_func 要求 float16 或 bfloat16
        output = flash_attn_varlen_func(
            q=q.to(torch.float16),
            k=k.to(torch.float16),
            v=v.to(torch.float16),
            cu_seqlens_q=context.cu_seqlens_q,  # 累积序列长度（Q 侧）
            cu_seqlens_k=context.cu_seqlens_k,  # 累积序列长度（K 侧）
            max_seqlen_q=context.max_seqlen_q,   # 最大 Q 序列长度
            max_seqlen_k=context.max_seqlen_k,   # 最大 K 序列长度
            softmax_scale=self.scale,            # 1/sqrt(head_dim)
            causal=True,                          # 因果掩码（只看当前位置及之前）
        )
        # 转回原始 dtype（可能是 bfloat16）
        return output.to(q.dtype)

    def _decode_attention(
        self,
        q: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Decode 阶段的 Attention：使用 flash_attn_with_kvcache。

        与 Prefill 的关键区别：
        - Q 只有 1 个 token（新生成的）
        - K/V 从 KV Cache 中读取（之前存储的整段历史）
        - 需要 block_tables 来定位物理页
        - 需要 context_lens（每个序列的当前长度）来知道该读多少历史
        """
        original_dtype = q.dtype

        # 从 Context 获取当前层的 KV Cache
        kv_cache = context.kv_cache[self.layer_idx]
        k_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache = kv_cache[1]

        # q: [num_seqs, num_heads, head_dim] → [num_seqs, 1, num_heads, head_dim]
        # flash_attn_with_kvcache 要求这个 shape
        q = q.unsqueeze(1).to(torch.float16)

        # cache_seqlens: 每个序列已经存储了多少个 token 的 KV（用来控制读取范围）
        cache_seqlens = context.context_lens.to(torch.int32)

        # block_table: 每个序列的逻辑块→物理页映射
        # shape: [num_seqs, max_num_blocks]
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

        # [num_seqs, 1, num_heads, head_dim] → [num_seqs, num_heads, head_dim]
        return output.squeeze(1).to(original_dtype)
```

---

## 5. ✅ 验证步骤

```bash
cd nano_vll_repro

# 运行 Day3 测试（Block + BlockManager + Attention）
python tests/test_Day3.py
```

> **⚠️ 现有 `tests/test_Day3.py` 有以下 bug，需要先修复再运行：**
>
> **Bug 1**：`test_attention_with_context()` 中 `set_context()` 传参方式错误（第 219-226 行）。
> ```python
> # ❌ 错误（当前代码）
> set_context(
>     is_prefill=True,
>     cu_seqlens_q=cu_seqlens,
>     ...
> )
>
> # ✅ 正确
> from utils.context import Context  # 确保顶部导入中包含 Context
> set_context(Context(
>     is_prefill=True,
>     cu_seqlens_q=cu_seqlens,
>     ...
> ))
> ```
> 同时需要在文件顶部的 import 中补上 `Context`：
> ```python
> # 当前（第 10 行）
> from utils.context import set_context, reset_context
> # 改为
> from utils.context import Context, set_context, reset_context
> ```
>
> **Bug 2**：`test_store_kvcache()` 中 `store_kvcache()` 调用签名错误（第 262 行）。
> ```python
> # ❌ 错误（当前代码）— 传了分离的 k_cache, v_cache 两个参数
> store_kvcache(key, value, k_cache, v_cache, slot_mapping)
>
> # ✅ 正确 — 应构造合并的 kv_cache tensor
> kv_cache = torch.stack([k_cache, v_cache], dim=0)  # [2, num_blocks, block_size, num_kv_heads, head_dim]
> store_kvcache(key, value, kv_cache, slot_mapping)
> ```
> `store_kvcache()` 的实际签名是 `(k, v, kv_cache, slot_mapping)`，`kv_cache` 是合并的 `[2, ...]` tensor。

如果看到 `🎉 Day 3 所有测试通过!`，说明 PagedAttention 引擎正常。

你也可以在 Python 中手动验证：

```python
import sys
sys.path.insert(0, '.')
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from sampling_params import SamplingParams

# 创建一个 BlockManager
bm = BlockManager(num_blocks=10, block_size=4)

# 创建序列：7 个 token，需要 2 个块（ceil(7/4) = 2）
seq = Sequence([1, 2, 3, 4, 5, 6, 7], SamplingParams())
seq.block_size = 4

# 分配
bm.allocate(seq)
print(f"block_table: {seq.block_table}")  # 例如 [0, 1]

# slot_mapping
slots = bm.get_slot_mapping(seq)
print(f"slot_mapping: {slots}")  # 例如 [0,1,2,3, 4,5,6]（取决于分配的物理页）
```

---

## 6. 📌 本篇核心记忆

三句话概括 PagedAttention：

1. **物理页（Block）= KV Cache 的最小管理单元**，大小固定（256 tokens），像操作系统的内存页
2. **block_table = 页表**，把逻辑块映射到物理页，可以是不连续的
3. **Prefix Cache = 共享内存**，相同内容的块被多个序列复用，通过引用计数管理生命周期

如果面试被问「PagedAttention 是什么」，标准回答：

> PagedAttention 是 vLLM 提出的一种 KV Cache 内存管理技术。它把 KV Cache 切成固定大小的 Block，像操作系统的虚拟内存一样管理。好处是：消除内部碎片和外部碎片，支持多个请求共享相同前缀（Prefix Cache），从而大幅提高 GPU 显存利用率和服务吞吐量。

---

下一篇：**Day4 — Qwen3 模型与权重加载**（GQA / QK Norm / 融合权重映射）
