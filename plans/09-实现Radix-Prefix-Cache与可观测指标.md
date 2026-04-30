# 09. 实现 Radix Prefix Cache 与可观测指标

这一篇继续吸收主线 `00~07` 之外的新能力，但仍然必须站在当前仓库的真实代码边界上。

这一篇只做三件事：

> 1. 把当前 `BlockManager` 里“完整块 + hash 表”的 prefix cache，升级成“完整块 + radix / prefix tree”的教学版实现。
> 2. 让 prefix cache 不只会复用，还会暴露命中、复用 token 数、块复用次数等可观测指标。
> 3. 保持它与当前 `Sequence / Scheduler / ModelRunner / Attention` 边界兼容。

这一篇不做下面这些事：

- 不引入新的 KV cache tensor 布局。
- 不改 `layers/attention.py` 的 Triton store kernel。
- 不把 prefix cache 扩展成跨进程共享服务。
- 不引入 speculative decoding。
- 不引入 offload。

原因很简单：

> 当前仓库最值得升级的，是“前缀复用的数据结构和观测能力”，而不是先把整条执行图推倒重来。

---

## 1. 为什么从 hash prefix cache 升级到 radix prefix cache

当前 `engine/block_manager.py` 的 prefix cache 已经能工作。它的核心思路是：

1. 只有完整块才计算 hash。
2. hash 里链入 `prefix_hash`，保证“相同前缀 + 相同当前块内容”才命中。
3. 命中后复用整个物理块。

这已经比“没有 prefix cache”强很多，但有两个限制。

### 1.1 它只能按完整块边界命中

两条序列的长前缀可能高度重合，但如果尾部重合不够一个完整 block，当前 hash 表无法表达“部分尾块复用”。这一篇先不做 partial-tail 复用，但会把前缀结构显式化，为后续扩展留出边界。

### 1.2 它的复用路径不透明

当前命中逻辑能工作，但很难回答：

- 命中了多少次 prefix cache？
- 一共复用了多少 token？
- 哪些块是通过 prefix 命中的？
- 当前 cache 的树形前缀覆盖情况大概是什么样？

radix / prefix-tree 方案的价值就在这里：

> 它不只是一个“能不能命中”的映射表，而是一种能表达“前缀共享结构”的数据结构。

---

## 2. 当前代码是什么状态

与 prefix cache 直接相关的地方主要有：

1. `engine/block_manager.py`
2. `engine/sequence.py`
3. `engine/scheduler.py`
4. `plans/02A-讲透PagedAttention、BlockManager与调度主线.md`

### 2.1 当前 `BlockManager` 里只有 `hash_to_block_id`

当前结构是：

```python
self.hash_to_block_id: dict[int, int] = {}
```

它适合做“这个完整块 hash 有没有出现过”，但不适合表达：

- 某个前缀下面还能继续接哪些块。
- 当前前缀树的深度和分支情况。
- 一条序列沿着哪条共享前缀路径匹配上。

### 2.2 当前 `Sequence` 已经有 `num_cached_tokens`

这很好，因为后面仍然可以继续用 `seq.num_cached_tokens` 作为“这条序列已经复用了多少前缀 token”的总账本。

### 2.3 当前系统还没有显式 observability 对象

所以需要一个轻量结构，专门统计：

- prefix cache hit 次数。
- miss 次数。
- reused blocks。
- reused tokens。

没有指标很容易产生错觉：代码看起来支持 prefix cache，但实际从来没命中过。

---

## 3. 本篇目标边界

当前仓库最稳妥的目标，是实现下面这套结构：

1. 块级 hash 仍然保留。
2. 在 hash 之上新增一棵 prefix tree。
3. 每个树节点代表“一个已经稳定的完整块”。
4. 父子关系代表“某个块在某个前缀之后继续延伸”。
5. 复用时优先沿着树和 hash 索引找稳定完整块。
6. 同时统计 prefix cache 观测指标。

换句话说：

> 这一篇做的是“块级 hash prefix cache 的结构升级版”，不是“任意 token 级压缩树”。

---

## 4. 修改 `engine/block_manager.py`

### 4.1 新增 prefix tree 和指标结构

先在 `engine/block_manager.py` 顶部增加两个轻量数据类。

```python
from dataclasses import dataclass, field


@dataclass
class PrefixCacheStats:
    """
    prefix cache 的基础可观测指标。

    这些指标只统计“块级前缀复用”这件事，
    不统计 decode append 或普通 block 分配。
    """
    hit_count: int = 0
    miss_count: int = 0
    reused_blocks: int = 0
    reused_tokens: int = 0
    inserted_blocks: int = 0

    def reset(self) -> None:
        self.hit_count = 0
        self.miss_count = 0
        self.reused_blocks = 0
        self.reused_tokens = 0
        self.inserted_blocks = 0


@dataclass
class PrefixTreeNode:
    """
    radix / prefix tree 的一个节点。

    教学版约定：
    - 一个节点对应一个已经稳定、已经填满、已经算出 hash 的完整 block。
    - root 节点本身不对应真实 block，只是树根。
    - children 的 key 直接使用当前块的 hash。
    """
    hash_value: int
    block_id: int | None
    token_ids: list[int] = field(default_factory=list)
    children: dict[int, "PrefixTreeNode"] = field(default_factory=dict)
    parent: "PrefixTreeNode | None" = None

    def is_root(self) -> bool:
        return self.block_id is None
```

### 4.2 在 `BlockManager.__init__()` 中接入新结构

把初始化逻辑里的 prefix cache 相关字段改成下面这样。

```python
# root 节点不对应任何真实 block。
self.prefix_tree_root = PrefixTreeNode(hash_value=-1, block_id=None)

# hash -> prefix tree node，用于快速定位完整块节点。
self.hash_to_node: dict[int, PrefixTreeNode] = {}

# 观测指标。
self.prefix_cache_stats = PrefixCacheStats()
```

### 4.3 新增树节点注册函数

```python
def _register_prefix_block(
    self,
    block_id: int,
    token_ids: list[int],
    current_hash: int,
    prefix_hash: int,
) -> None:
    """
    把一个已经填满、已经稳定的 block 注册到 prefix tree。

    参数含义：
    - block_id: 当前物理块 ID。
    - token_ids: 当前完整块中的 token 列表。
    - current_hash: 当前块在 prefix-aware 条件下计算出的 hash。
    - prefix_hash: 父前缀块的 hash；-1 表示挂到 root 下。
    """
    block = self.blocks[block_id]
    block.update(current_hash, token_ids.copy())

    if prefix_hash == -1:
        parent = self.prefix_tree_root
    else:
        parent = self.hash_to_node[prefix_hash]

    node = self.hash_to_node.get(current_hash)
    if node is None:
        node = PrefixTreeNode(
            hash_value=current_hash,
            block_id=block_id,
            token_ids=token_ids.copy(),
            parent=parent,
        )
        self.hash_to_node[current_hash] = node
        self.prefix_cache_stats.inserted_blocks += 1

    parent.children[current_hash] = node
```

### 4.4 新增基于 prefix tree 的命中检查

```python
def _lookup_prefix_block(
    self,
    token_ids: list[int],
    prefix_hash: int,
) -> tuple[int, int]:
    """
    沿着 prefix tree 查找当前完整块是否已经存在。

    返回：
    - 命中时返回 (block_id, current_hash)。
    - 未命中时返回 (-1, current_hash)。
    """
    current_hash = self.compute_hash(token_ids, prefix_hash)
    node = self.hash_to_node.get(current_hash)

    if node is None:
        self.prefix_cache_stats.miss_count += 1
        return -1, current_hash

    if node.token_ids != token_ids:
        self.prefix_cache_stats.miss_count += 1
        return -1, current_hash

    self.prefix_cache_stats.hit_count += 1
    self.prefix_cache_stats.reused_blocks += 1
    self.prefix_cache_stats.reused_tokens += len(token_ids)
    return int(node.block_id), current_hash
```

### 4.5 替换 `allocate()`

下面给出完整教学版实现。

```python
def allocate(self, seq: Sequence):
    """
    为序列分配完整 prompt 需要的逻辑块。

    这一版与旧版最大的区别是：
    - 命中检查不再只是 hash_to_block_id。
    - 命中路径同时会把块视为 prefix tree 上的一段前缀路径。
    - 命中 / miss 会更新可观测指标。
    """
    assert not seq.block_table, "Sequence already has blocks allocated"

    prefix_hash = -1

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        is_full_block = len(token_ids) == self.block_size

        if is_full_block:
            cached_block_id, current_hash = self._lookup_prefix_block(token_ids, prefix_hash)

            if cached_block_id != -1:
                cached_block = self.blocks[cached_block_id]
                seq.num_cached_tokens += self.block_size

                if cached_block_id in self.used_block_ids:
                    cached_block.ref_count += 1
                else:
                    self._recover_block(cached_block_id)

                seq.block_table.append(cached_block_id)
                prefix_hash = current_hash
                continue

            fresh_block = self._allocate_fresh_block()
            block_id = fresh_block.block_id
            self._register_prefix_block(block_id, token_ids, current_hash, prefix_hash)
            seq.block_table.append(block_id)
            prefix_hash = current_hash
            continue

        fresh_block = self._allocate_fresh_block()
        seq.block_table.append(fresh_block.block_id)
```

### 4.6 替换 `append_slot()`

chunked prefill、decode 以及后续新生成 token 都可能让“原本不完整的末尾块”在某一刻刚好填满。`append_slot()` 在“当前块刚好填满”时，也必须把它挂进 prefix tree。

```python
def append_slot(self, seq: Sequence):
    """
    为 decode 新 token 分配 slot。

    三种情况：
    1. 刚好需要新 block：分配 fresh block。
    2. 刚好填满当前 block：把这个完整块登记到 prefix tree。
    3. 其他：无需操作。
    """
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    position_in_block = len(seq) % self.block_size

    if position_in_block == 1:
        assert last_block.hash != -1, "Previous block should be complete"
        new_block = self._allocate_fresh_block()
        block_table.append(new_block.block_id)

    elif position_in_block == 0:
        assert last_block.hash == -1, "Block already has hash"
        token_ids = seq.block(seq.num_blocks - 1)
        prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        current_hash = self.compute_hash(token_ids, prefix_hash)
        self._register_prefix_block(last_block.block_id, token_ids, current_hash, prefix_hash)
```

### 4.7 新增观测接口

```python
def get_prefix_cache_stats(self) -> dict:
    """
    返回 prefix cache 的快照指标。

    这个方法只负责导出事实，不负责打印格式。
    benchmark、日志或更上层的监控脚本都可以复用它。
    """
    return {
        "hit_count": self.prefix_cache_stats.hit_count,
        "miss_count": self.prefix_cache_stats.miss_count,
        "reused_blocks": self.prefix_cache_stats.reused_blocks,
        "reused_tokens": self.prefix_cache_stats.reused_tokens,
        "inserted_blocks": self.prefix_cache_stats.inserted_blocks,
    }


def reset_prefix_cache_stats(self) -> None:
    """
    清空当前统计。
    """
    self.prefix_cache_stats.reset()
```

---

## 5. 为什么这一版仍然与当前仓库兼容

这一版看起来变化很大，但它故意没有碰下面这些稳定边界：

1. `Sequence.block_table` 仍然是“逻辑块 -> 物理块”的映射。
2. `slot_mapping` 的计算方式不变。
3. `Attention` 仍然只关心真实 `slot_mapping` 和 `kv_cache`。
4. `Scheduler` 仍然只和“能不能 allocate / can_append / deallocate”打交道。

也就是说：

> 这一篇改的是 prefix cache 的“命中结构”和“观测能力”，不是把整个 PagedAttention 主线推翻重来。

---

## 6. 新增 `tests/test_Day9_radix_cache.py`

这份测试不跑大模型，专门锁住三类边界：

1. prefix tree 节点会被正确登记。
2. 同前缀完整块会命中复用。
3. 指标会正确增长。

```python
"""Day9 radix prefix cache 结构测试。"""

import sys

sys.path.insert(0, ".")

from engine.sequence import Sequence
from engine.block_manager import BlockManager
from sampling_params import SamplingParams


def test_prefix_cache_stats_export():
    manager = BlockManager(num_blocks=8, block_size=4)
    stats = manager.get_prefix_cache_stats()

    assert stats["hit_count"] == 0
    assert stats["miss_count"] == 0
    assert stats["reused_blocks"] == 0
    assert stats["reused_tokens"] == 0
    assert stats["inserted_blocks"] == 0


def test_full_block_is_registered_into_prefix_tree():
    manager = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([1, 2, 3, 4], SamplingParams())
    seq.block_size = 4

    manager.allocate(seq)

    stats = manager.get_prefix_cache_stats()
    assert stats["inserted_blocks"] == 1
    assert len(seq.block_table) == 1

    root_children = manager.prefix_tree_root.children
    assert len(root_children) == 1


def test_same_full_block_hits_prefix_cache():
    manager = BlockManager(num_blocks=8, block_size=4)

    seq1 = Sequence([1, 2, 3, 4], SamplingParams())
    seq2 = Sequence([1, 2, 3, 4], SamplingParams())
    seq1.block_size = 4
    seq2.block_size = 4

    manager.allocate(seq1)
    manager.allocate(seq2)

    stats = manager.get_prefix_cache_stats()
    assert stats["hit_count"] >= 1
    assert stats["reused_blocks"] >= 1
    assert stats["reused_tokens"] >= 4
    assert seq1.block_table[0] == seq2.block_table[0]


def test_partial_last_block_is_not_registered_as_prefix_node():
    manager = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([10, 11, 12, 13, 14], SamplingParams())
    seq.block_size = 4

    manager.allocate(seq)

    stats = manager.get_prefix_cache_stats()
    assert stats["inserted_blocks"] == 1
    assert len(seq.block_table) == 2
```

---

## 7. 验收命令

```bash
python -m py_compile engine/block_manager.py tests/test_Day9_radix_cache.py
python tests/test_Day9_radix_cache.py
```

如果你想做一轮轻量手测，可以再跑：

```bash
python - <<'PY'
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from sampling_params import SamplingParams

manager = BlockManager(num_blocks=16, block_size=4)

seq1 = Sequence([1, 2, 3, 4], SamplingParams())
seq2 = Sequence([1, 2, 3, 4], SamplingParams())
seq1.block_size = 4
seq2.block_size = 4

manager.allocate(seq1)
manager.allocate(seq2)

print("seq1 blocks:", seq1.block_table)
print("seq2 blocks:", seq2.block_table)
print("stats:", manager.get_prefix_cache_stats())
PY
```

如果实现正确，你应该能看到：

- `seq1` 和 `seq2` 的第一块映射到同一个物理块。
- `stats` 里 `hit_count`、`reused_blocks`、`reused_tokens` 都大于 0。

---

## 8. 常见坑

1. **把 radix tree 理解成“任意 token 级压缩 trie”，然后试图重写整个 KV cache 地址体系。**
   这一篇完全不需要走那么远。当前教学版只需要做完整块级前缀树。
2. **删除 hash 检查，只保留 tree。**
   这样会让快速命中索引能力变差。教学版最稳妥的是“tree 表达结构，hash 做快速索引”。
3. **最后一个不完整块也注册进 prefix cache。**
   这样很容易把不稳定尾块错误复用出去。
4. **只做复用，不做观测指标。**
   这会让你根本不知道 prefix cache 到底有没有起作用。
5. **把 prefix tree 节点直接塞进 `Sequence` 里长期保存。**
   当前教学仓库没必要让序列持有这么重的状态；统计和复用逻辑都放在 `BlockManager` 更稳。

---

## 9. 本篇结束后你应该明白

这一篇最重要的不是“会写一个树节点类”。

真正要学会的是：

1. prefix cache 的升级重点，不只是“再快一点”，而是“更准确地表达前缀共享结构”。
2. 当前仓库最适合的升级路径，是在现有 block / hash 体系上叠加 prefix tree，而不是改 KV cache tensor 布局。
3. observability 不是附属品，而是判断 prefix cache 是否真的有价值的必要条件。
4. 这一步做完以后，后面的 speculative decoding、MoE 和 offload 才更容易建立在“看得见账本”的推理系统上。

下一篇进入 speculative decoding：

- `10-实现Speculative-Decoding基础版.md`
