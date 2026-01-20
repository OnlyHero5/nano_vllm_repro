import torch
from torch import nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: 
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, 
                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                  slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1  # 新增
    assert k_cache.stride(1) == D and v_cache.stride(1) == D    # 新增！关键
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), 
        value, value.stride(0), 
        k_cache, v_cache,
        slot_mapping, D
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        layer_idx: int = 0,  # 可以保留但不再使用
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 关键：添加这两个属性，由 model_runner 赋值
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        
        k_cache, v_cache = self.k_cache, self.v_cache
        
         # 存储 KV 到 cache
        if k_cache.numel() and v_cache.numel():
            # [修复代码 START] 
            # 必须显式转换为 Cache 的数据类型 (float16)，否则 triton 写入会出错导致乱码
            k_to_cache = k.to(k_cache.dtype)
            v_to_cache = v.to(v_cache.dtype)
            store_kvcache(k_to_cache, v_to_cache, k_cache, v_cache, context.slot_mapping)            # [修复代码 END]
            
        if context.is_prefill:
            # DEBUG
            print(f"[Attention] Prefill context: cu_seqlens_q={context.cu_seqlens_q}")
            
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
        else:
            # Decode
            q_expanded = q.unsqueeze(1).to(torch.float16)
            output = flash_attn_with_kvcache(
                q=q_expanded,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
            return output.squeeze(1).to(q.dtype)