from dataclasses import dataclass
import torch

from typing import Optional


@dataclass
class Context:
    """
    全局推理上下文

    在模型前向传播过程中， Attention层需要额外的元数据来正确处理 KV Cache

    使用全局 Context 避免在模型各层之间传递大量参数。
    """

    # ===阶段标识===
    is_prefill: bool = False

    # ===Prefill阶段参数（FlashAttention varlen API 需要）===
    cu_seqlens_q: torch.Tensor | None = None    # Query累积序列长度
    cu_seqlens_k: torch.Tensor | None = None    # Key累积序列长度
    max_seqlen_q: int = 0   # Query 最大序列长度
    max_seqlen_k: int = 0   # Key 最大序列长度

    # ===KV Cache 写入参数===
    slot_mapping: torch.Tensor | None = None    # token -> cache slot 映射

    # ===Decode===
    context_lens: torch.Tensor | None = None    # 每个序列的上下文长度 [num_seqs]
    block_tables: torch.Tensor | None = None    # 所有序列的block_size [num_seqs, max_blocks]
    max_context_len: int = None # 最大上下文长度
    max_num_blocks: int = None # 最大块数

    # ===== KV Cache 引用 =====
    kv_cache: Optional[list[torch.Tensor]] = None # [num_layers] 每层的KV cache

# 全局单例
_current_context = Context()


def get_context() -> Context:
    """获取当前上下文"""
    global _current_context
    if _current_context is None:
        raise RuntimeError("Context not set. Call set_context() first")



def set_context(context: Context):
    "设置当前上下文"
    global _current_context
    _current_context = context


def clear_context():
    """清除当前上下文"""
    global _CONTEXT
    _CONTEXT = Context()