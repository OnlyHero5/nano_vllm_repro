from dataclasses import dataclass
import torch



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
    context_lens: torch.Tensor | None = None    # 每个序列的上下文长度
    block_tables: torch.Tensor | None = None    # 所有序列的block_size

# 全局单例
_CONTEXT = Context()


def get_context() -> Context:
    """获取当前上下文"""
    return _CONTEXT



def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
):
    """设置上下文（每次推理前调用）"""
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables
    )



def reset_context():
    """重置上下文（推理结束后）"""
    global _CONTEXT
    _CONTEXT = Context()