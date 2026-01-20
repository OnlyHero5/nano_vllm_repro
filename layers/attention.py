"""PagedAttention
整合 Triton store_kvcache kernel 和 FlashAttention。
从全局 Context 获取 KV Cache 引用和元数据。

两种模式：
- Prefill: flash_attn_varlen_func（处理完整 prompt）
- Decode: flash_attn_with_kvcache（逐 token 生成）
"""
import torch
from torch import nn

from utils.context import get_context, Context

import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

@triton.jit
def store_kvcache_kernel(
    K, V,
    KCache, VCache,
    slot_mapping,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_kcb, stride_kcs, stride_kch,stride_kcd,
    stride_vcb, stride_vcs, stride_vch,stride_vcd,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_H: tl.constexpr, #分块计算参数，一次处理多少head, num_heads
    BLOCK_D: tl.constexpr  #分块计算参数，一次处理多少维度, head_dim
):
    """将K, V 写入KV Cache 指定的slot
       每个 program 处理一个token。
    """

    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping + token_idx)

    # 计算 block_id 和 block_offset
    block_id = slot // block_size
    block_offset = slot % block_size

    # 遍历所有head 和 head_dim
    for h in range(0, num_heads, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < num_heads  # 防止越界

        for d in range(0, head_dim, BLOCK_D):
            d_offsets = d + tl.arange(0, BLOCK_D)
            d_mask = d_offsets  < head_dim # 防止越界

            mask = h_mask[:, None] & d_mask[None, :]

            # 读取K V 
            k_ptrs = K + token_idx * stride_kn + h_offsets[:, None] * stride_kh + d_offsets[None, :] * stride_kd
            v_ptrs = V + token_idx * stride_vn + h_offsets[:, None] * stride_vh + d_offsets[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=mask, other=0.0)
            v = tl.load(v_ptrs, mask=mask, other=0.0)

            # 写入Cache
            kc_ptrs = KCache + block_id * stride_kcb + block_offset * stride_kcs + h_offsets[:, None] * stride_kch + d_offsets[None, :] * stride_kcd
            vc_ptrs = VCache + block_id * stride_vcb + block_offset * stride_vcs + h_offsets[:, None] * stride_vch + d_offsets[None, :] * stride_vcd

            tl.store(kc_ptrs, k, mask=mask)
            tl.store(vc_ptrs, v, mask=mask)

def store_kvcache(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor
):
    """将 K, V 存入 KV Cache
    
    Args:
        k: [num_tokens, num_kv_heads, head_dim]
        v: [num_tokens, num_kv_heads, head_dim]
        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] 每个 token 的 slot
    """
    num_tokens, num_heads, head_dim = k.shape
    block_size = kv_cache.shape[2]

    # KCache, VCache
    k_cache = kv_cache[0]   # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache = kv_cache[1]

    # 确保连续内存
    k = k.contiguous()
    v = v.contiguous()

    # 启动 kernel
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



class Attention(nn.Module):
    """PagedAttention with FlashAttention
    
    支持两种模式：
    - Prefill: 使用 flash_attn_varlen_func
    - Decode: 使用 flash_attn_with_kvcache
    """
    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            scale: float,
            num_kv_heads: int,
            layer_idx: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q: [num_tokens, num_heads, head_dim]
            k: [num_tokens, num_kv_heads, head_dim]
            v: [num_tokens, num_kv_heads, head_dim]
        
        Returns:
            [num_tokens, num_heads, head_dim]
        """
        context = get_context()

        # 存储K V 到 cache
        if context.kv_cache is not None and context.slot_mapping is not None:
            store_kvcache(
                k, v,
                context.kv_cache[self.layer_idx],
                context.slot_mapping
            )
        if context.is_prefill:
            return self._prefill_attention(q, k, v, context)
        else:
            return self._decode_attention(q, context)
    
    def _prefill_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            context: Context,
    ) -> torch.Tensor:
        """Prefill: 使用 flash_attn_varlen_func"""
        output = flash_attn_varlen_func(
            q=q.to(torch.float16),
            k=k.to(torch.float16),
            v=v.to(torch.float16),
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True
        )
        return output.to(q.dtype)
    
    def _decode_attention(
            self,
            q: torch.Tensor,
            context: Context
    ) -> torch.Tensor:
        """Decode: 使用 flash_attn_with_kvcache"""
        original_dtype = q.dtype
        kv_cache = context.kv_cache[self.layer_idx]
        k_cache = kv_cache[0] # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache = kv_cache[1]

        # q: [num_seqs, num_heads, head_dim] -> [num_seqs, 1, num_heads, head_dim]
        q = q.unsqueeze(1).to(torch.float16)

        output = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True
        )

        # [num_seqs, 1, num_heads, head_dim] -> [num_seqs, num_heads, head_dim]
        return output.squeeze(1).to(original_dtype)
    
    
