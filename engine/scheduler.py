"""Scheduler - Continuous Batching 调度器

核心职责：
1. 管理 waiting/running 两个队列
2. 决定每个 step 处理哪些序列
3. 与 BlockManager 协作管理 KV Cache
4. 处理序列完成和 preemption

调度策略：
- Prefill 优先：新请求优先处理
- FCFS (First Come First Serve): 先到先服务
- Preemption: 内存不足时抢占最后加入的序列
"""
from collections import deque
from typing import Tuple, List

from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager

class Scheduler:
    """Continuous Batching 调度器
    
    管理请求的生命周期：
    WAITING → RUNNING → FINISHED
    
    Attributes:
        waiting: 等待 Prefill 的序列队列
        running: 正在 Decode 的序列队列
        block_manager: KV Cache 块管理器
    """
    def __init__(self, config: Config, block_manager: BlockManager):
        """Continuous Batching 调度器
    
        管理请求的生命周期：
        WAITING → RUNNING → FINISHED
    
        Attributes:
            waiting: 等待 Prefill 的序列队列
            running: 正在 Decode 的序列队列
            block_manager: KV Cache 块管理器
        """

        # 配置参数
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        # Block Manager
        self.block_manager = block_manager

        # 双队列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """检查请求是否都已完成"""
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, seq: Sequence):
        """添加新序列到 waiting 队列
        
        Args:
            seq: 新的序列（状态为 WAITING）
        """
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)
    
    add = add_sequence # 别名
    
    def schedule(self) -> Tuple[List[Sequence], bool]:
        """核心调度方法
        
        决定当前 step 处理哪些序列。
        
        Returns:
            (scheduled_seqs, is_prefill)
            - scheduled_seqs: 本次要处理的序列列表
            - is_prefill: True 表示 Prefill 阶段，False 表示 Decode 阶段
        """
        scheduled_seqs: List[Sequence] = []
        num_seqs = 0  #记录当前 Batch 里已经塞进了多少个序列
        num_batched_tokens = 0  #记录当前 Batch 里所有序列的 Token 总数。

        # ===== 阶段1：尝试prefill =====
        # 优先处理 waiting队列 中的新请求
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # 检查 Token数量约束
            # num_cached_tokens 是 prefix cache的命中部分。不需要重新计算
            new_tokens = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + new_tokens > self.max_num_batched_tokens:
                break

            # 检查KV cache是否足够
            if not self.block_manager.can_allocate(seq):
                break

            # 分配blocks并移动到running队列
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

            num_seqs += 1
            num_batched_tokens += new_tokens
        
        # 如果有prefill队列，返回
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # ===== 阶段2：Decode =====
        # 处理running队列中正在生成的序列
        # 临时存储本轮处理的序列
        decoded_seqs: List[Sequence] = []

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # 检查是否可以追加slot
            while not self.block_manager.can_append(seq):
                # KV cache 不足，需要preempt
                if self.running:
                    # 抢占最后加入的序列（LRU策略）
                    victim = self.running.pop()
                    self.__preempt(victim)
                else:
                    # 没有其他序列抢占，只能抢占当前序列，防止卡死
                    self.__preempt(seq)
                    break
            else:
                # 成功获取slot
                self.block_manager.append_slot(seq)
                decoded_seqs.append(seq)
                num_seqs += 1
        
        # 将处理的序列放回running队列头部
        for seq in reversed(decoded_seqs):
            self.running.appendleft(seq)
        
        return decoded_seqs, False

    def __preempt(self, seq: Sequence):
        """抢占序列（释放其 KV Cache）
        
        当 KV Cache 不足时，将序列移回 waiting 队列。
        下次调度时需要重新 Prefill。
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        # 放到waiting队列最前面(优先重新计算)
        self.waiting.appendleft(seq)

    def postprocess(
            self,
            seqs: List[Sequence],
            token_ids: List[int],
    ) -> List[Sequence]:
        """后处理：更新序列状态
        
        在模型 forward 和采样之后调用：
        1. 将生成的 token 追加到序列
        2. 检查终止条件
        3. 释放已完成序列的资源
        
        Args:
            seqs: 本轮处理的序列
            token_ids: 生成的 token ID 列表
        
        Returns:
            已完成的序列列表
        """
        finished_seqs: List[Sequence] = []

        for seq, token_id in zip(seqs, token_ids):
            # 追加 新token
            seq.append_token(token_id)

            # 检查终止条件
            is_eos = (not seq.ignore_eos) and (token_id == self.eos)
            is_max_tokens = seq.num_completion_tokens >= seq.max_tokens

            if is_eos or is_max_tokens:
                # 序列完成
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_seqs.append(seq)
        
        return finished_seqs
    
    def get_num_waiting(self) -> int:
        """获取等待队列长度"""
        return len(self.waiting)
    
    def get_num_running(self) -> int:
        """获取运行队列长度"""
        return len(self.running)
    
    def __repr__(self) -> str:
        return (f"Scheduler(waiting = {self.get_num_waiting()},"
                f"running={self.get_num_running()},"
                f"free_blocks={self.block_manager.get_num_free_blocks()})")