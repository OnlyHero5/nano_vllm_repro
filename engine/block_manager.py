"""Block Manager - PagedAttention 核心组件

实现类似操作系统分页的 KV Cache 管理：
- Block：物理内存块，固定大小（默认 256 tokens）
- BlockManager：管理 block 的分配、释放、复用

关键概念：
- 物理块(Physical Block)：GPU 显存中实际存储 KV 的位置
- 逻辑块(Logical Block)：序列视角看到的块索引
- block_table：逻辑块 → 物理块的映射表
- slot_mapping：token 位置 → 物理 cache 槽位的映射
"""

from collections import deque
from engine.sequence import Sequence

try:
    import xxhash
    import numpy as np
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    print("[警告] xxhash not installed, using builin hash (slower)")

class Block:
    """物理内存块
    每个 Block 可存储 block_size 个 token 的 KV Cache。
    
    Attributes:
        block_id: 物理块 ID（在 kv_cache tensor 中的索引）
        ref_count: 引用计数（支持多序列共享，如 beam search）
        hash: 内容哈希（用于 Prefix Caching）
        token_ids: 存储的 token ID 列表（用于验证 cache 命中）
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1          # -1 表示未完成/无效
        self.token_ids = []     # 用于prefix cache 验证
    
    def update(self, hash_value: int, token_ids: list[int]):
        """更新 block 的哈希 和 内容 （block填满时）"""
        self.hash = hash_value
        self.token_ids = token_ids
    
    def reset(self):
        """重置block， 重新分配时"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    
    def __repr__(self):
        return f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash})"



class BlockManager:
    """Block 管理器
    
    核心职责：
    1. 维护空闲/已用 block 池
    2. 为序列分配/释放 blocks
    3. 支持 Prefix Caching（基于内容哈希复用 block）
    
    数据结构：
    - blocks: 所有物理 block 的列表
    - free_block_ids: 空闲 block ID 队列
    - used_block_ids: 已用 block ID 集合
    - hash_to_block_id: 哈希 → block ID 映射（Prefix Cache）
    """
    
    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 总 block 数（根据 GPU 显存计算）
            block_size: 每个 block 存储的 token 数（默认 256）
        """
        self.block_size = block_size
        self.num_blocks = num_blocks

        # 创建所有blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 空闲队列 (deque高效两端操作)
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 已用集合
        self.used_block_ids: set[int] = set()

        # Prefix Cache 映射： hash -> block_id
        self.hash_to_block_id: dict[int, int] = {}

    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """计算 token 序列的哈希值
        
        支持链式哈希：当前块的哈希依赖于前一块的哈希，
        确保相同前缀的序列能够匹配。
        
        Args:
            token_ids: 当前块的 token 列表
            prefix_hash: 前一块的哈希值（-1 表示无前缀）
        
        Returns:
            64-bit 哈希值
        """
        if HAS_XXHASH:
            h = xxhash.xxh64()
            if prefix_hash != -1:
                h.update(prefix_hash.to_bytes(8, "little"))
            h.update(np.array(token_ids, dtype=np.int64).tobytes())
            return h.intdigest()
        else:
            # 使用内置hash
            return hash((prefix_hash, tuple(token_ids)))
        
    def _allocate_block(self, block_id: int) -> Block:
        """内部方法：分配一个全新的 block（用于 Cache Miss）
        
        区别：
        ✅ 调用 reset() -> 彻底清空旧数据
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already in use"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _recover_block(self, block_id: int) -> Block:
        """内部方法：复活空闲池中的指定 block（用于 Cache 命中）
        
        与 _allocate_block 的区别：
        ❌ 不调用 reset() -> 保留原有的 hash 和 token_ids
        ✅ 仅恢复 ref_count 和 pool 状态
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already in use"

        # 从空闲池里捞出
        self.free_block_ids.remove(block_id) # O(n)
        self.used_block_ids.add(block_id)
        block.ref_count = 1
        return block
    
    def _allocate_fresh_block(self) -> Block:
        """内部方法：从空闲池头部取出一个新块（用于 Cache Miss）
        
        性能：O(1) - 使用 popleft
        动作：调用 reset()，彻底清空旧数据
        """
        if not self.free_block_ids:
            raise ValueError("Out of memory! No free blocks available.")
            
        # 关键优化：内部决定用哪一块，而不是外部传入
        block_id = self.free_block_ids.popleft() 
        
        block = self.blocks[block_id]
        # 防御性断言
        assert block.ref_count == 0, f"Block {block_id} is already in use"
        
        block.reset() # 彻底清空，因为是新分配的
        
        self.used_block_ids.add(block_id)
        block.ref_count = 1
        
        return block
    
    def _deallocate_block(self, block_id: int):
        """释放指定block"""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} still has references"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_num_free_blocks(self) -> int:
        """获取空闲block数量"""
        return len(self.free_block_ids)
    
    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲blocks分配序列 悲观检查（真实vllm预先计算命中率）"""
        return len(self.free_block_ids) >= seq.num_blocks
    
    def allocate(self, seq: Sequence):
        """为序列分配所有需要的 blocks（Prefill 阶段）
        
        支持 Prefix Caching：
        1. 计算每个块的哈希
        2. 检查是否有现有 block 命中
        3. 命中则复用（增加引用计数），否则分配新 block
        
        Args:
            seq: 要分配的序列
        """
        assert not seq.block_table, "Sequence already has blocks allocated"

        prefix_hash = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            # 获取当前块的tokens
            token_ids = seq.block(i)

            # 只有完整的块才计算hash
            is_full_block = (len(token_ids) == self.block_size)
            current_hash = self.compute_hash(token_ids, prefix_hash) if is_full_block and not cache_miss else -1

            # 从prefix cache查找
            cached_block_id = self.hash_to_block_id.get(current_hash, -1)

            # =====分支A:验证缓存命中(哈希可能冲突，需要验证内容)=====
            if cached_block_id != -1:
                cached_block = self.blocks[cached_block_id]
                if cached_block.token_ids == token_ids:
                    # Cache命中
                    seq.num_cached_tokens += self.block_size
                    if cached_block_id in self.used_block_ids:
                        # 已经在被使用区
                        cached_block.ref_count += 1
                    else:
                        # block在空闲池
                        # 复活空闲块
                        # self._allocate_block(cached_block_id) (疑似优化问题)
                        self._recover_block(cached_block_id)
                    
                    seq.block_table.append(cached_block_id)
                    prefix_hash = current_hash
                    continue
            
            # ===== 分支B：未命中 ======
            cache_miss = True
            # block_id = self.free_block_ids[0]
            # block = self._allocate_block(block_id)
            block = self._allocate_fresh_block()
            block_id = block.block_id

            # 如果是完整块，更新缓存
            if is_full_block:
                if current_hash == -1:
                    current_hash = self.compute_hash(token_ids, prefix_hash)
                block.update(current_hash, token_ids.copy())
                self.hash_to_block_id[current_hash] = block_id
                prefix_hash = current_hash
            
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有 blocks
        
        引用计数减一，只有当计数归零时才真正释放。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1

            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    
    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以追加 token（Decode 阶段）
        
        只有当需要新 block 时才检查空闲池。
        需要新 block 的条件：序列长度 % block_size == 1（刚好超出上一个块）
        """
        needs_new_block = (len(seq) % self.block_size == 1)
        return len(self.free_block_ids) >= needs_new_block

    def append_slot(self, seq: Sequence):
        """为新生成的 token 分配 slot（Decode 阶段）
        
        三种情况：
        1. 刚好需要新 block：分配一个新 block
        2. 刚好填满上一个 block：更新该 block 的哈希
        3. 其他：无需操作（slot 在已分配的 block 内）
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        position_in_block = len(seq) % self.block_size

        if position_in_block == 1:
            # 情况1：刚超出上一个block，需要分配新的block
            # 此时上一个block应当已有有效的hash
            assert last_block.hash != -1, "Previous block should be complete"

            new_block = self._allocate_fresh_block()
            block_table.append(new_block.block_id)

        elif position_in_block == 0:
            # 情况2：刚好填满当前block
            # 计算哈希并加入 Prefix Cache
            assert last_block.hash == -1, "Block already has hash"

            token_ids = seq.block(seq.num_blocks - 1)
            prefix_hash = self.blocks[block_table[-2]].hash if len(block_table) > 2 else -1
            current_hash = self.compute_hash(token_ids, prefix_hash)

            last_block.update(current_hash, token_ids.copy())
            self.hash_to_block_id[current_hash] = last_block.block_id
        
        # 情况 3：position_in_block  在 2 到 block_size -1 之间
        # 无需操作，slot 在当前block内
    
    def get_slot_mapping(self, seq: Sequence, start_pos: int = 0) -> list[int]:
        """计算 slot mapping（从 start_pos 开始的所有 token）
        
        Args:
            seq: 序列
            start_pos: 起始位置（Prefill 时为 0，Decode 时为之前的长度）
        
        Returns:
            slot 列表，每个 token 对应一个 slot
        """
        slots = []
        for pos in range(start_pos, len(seq)):
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = seq.block_table[block_idx]
            slot = block_id * self.block_size + offset
            slots.append(slot)
        return slots

    def __repr__(self):
        return (f"BlockManager(num_blocks={self.num_blocks}, "
                f"free={len(self.free_block_ids)}, "
                f"used={len(self.used_block_ids)}")
    