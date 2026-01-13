"""Attention 层 - 集成 FlashAttention 和 PagedAttention

核心功能：
1. store_kvcache: 将 K/V 写入 paged cache
2. Prefill: 使用 flash_attn_varlen_func 处理变长序列
3. Decode: 使用 flash_attn_with_kvcache 从 paged cache 读取

依赖：
- flash_attn 库（需要安装）
- triton（用于自定义 kernel）
"""
import torch
from torch import nn

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("[警告] 未安装 flash-attn")

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[警告] triton 未安装")

from utils.context import get_context

# =====KV Cache 存储 =====
if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_prt,
        D: tl.constexpr
    ):
        """Triton kernel: 将 K/V 写入对应的 cache slot
        
        每个 program 处理一个 token 的 K/V。
        """

        idx = tl.program_id(0)
        # 加载slot映射
        slot = tl.load(slot_mapping_prt + idx)

        # slot = -1 表示跳过 (用于padding)
        if slot == -1:
            return
        
        # 计算偏移量
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)

        # 加载 K/V
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)

        # 写入 cache
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)



def store_kvcache_trition(
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor
):
    """使用 Triton 高效存储 KV Cache
    
    Args:
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] - 每个 token 对应的 slot
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim

    # 验证内存布局
    assert key.stride(-1) == 1 and value.stride(-1) == 1, "K/V must be contiguous in last dim"
    assert key.stride(1) == head_dim and value.stride(1) == head_dim

    # 启动 kernel
    store_kvcache_kernel[(num_tokens,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache.view(-1, D),
        v_cache.view(-1, D),
        slot_mapping,
        D
    )


# triton替代品
def store_kvcache_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    """PyTorch 版本的 KV Cache 存储（作为 fallback）
    
    效率较低但不需要 Triton。
    """

    # 展平cache为[total_slots, num_kv_heads, head_dim]
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
    v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)
    
    # 过滤有效 slots
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    
    # 写入 cache
    k_cache_flat[valid_slots] = key[valid_mask]
    v_cache_flat[valid_slots] = value[valid_mask]

store_kvcache = store_kvcache_trition if HAS_TRITON else store_kvcache_pytorch






# =============Attention================
class Attention(nn.Module):
    """通用 Attention 层（支持 Prefill 和 Decode）
    
    通过全局 Context 获取当前阶段的元数据：
    - Prefill: 使用 flash_attn_varlen_func
    - Decode: 使用 flash_attn_with_kvcache
    
    Attributes:
        k_cache: K cache tensor（由 ModelRunner 设置）
        v_cache: V cache tensor（由 ModelRunner 设置）
    """

    def __init__(
            self,
            num_heads: int,
            head_dim: int,
            scale: int,
            num_kv_heads: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # KV Cache 占位符
        self.k_cache: torch.Tensor = torch.empty(0)
        self.v_cache: torch.Tensor = torch.empty(0)
    
    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q: Query [num_tokens, num_heads, head_dim]
            k: Key [num_tokens, num_kv_heads, head_dim]
            v: Value [num_tokens, num_kv_heads, head_dim]
        
        Returns:
            output: [num_tokens, num_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        #===== 存储KV 到 cache =====
        if k_cache.numel() > 0 and v_cache.numel() > 0:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        #===== 根据阶段选择 attention 实现 =====
        if context.is_prefill:
            output = self._prefill_attention(q, k, v, k_cache, v_cache, context)
        else:
            output = self._decode_attention(q, k, v, k_cache, v_cache, context)
        
        return output


    def _prefill_attention(self, q, k, v, k_cache, v_cache, context) -> torch.Tensor:
        """Prefill 阶段的 Attention
        
        使用 flash_attn_varlen_func 处理变长序列。
        如果有 Prefix Cache 命中，从 cache 读取 K/V。
        """
        if HAS_FLASH_ATTN:
            # 检查是否有Prefix Cache
            if context.block_tables is not None and k_cache.numel() > 0:
                # 从cache中读取
                k_input, v_input = k_cache, v_cache
            else:
                k_input, v_input = k, v
            
            output = flash_attn_varlen_func(
                q, k_input, v_input,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            output = self._pytorch_attention(q, k, v, context)
        
        return output


        
    def _decode_attention(self, q, k, v, k_cache, v_cache, context) -> torch.Tensor:
        """Decode 阶段的 Attention
        
        使用 flash_attn_with_kvcache 直接从 paged cache 读取。
        """
        if HAS_FLASH_ATTN:
            # Decode 时 q 的 seq_len = 1，需要 unsqueeze
            output = flash_attn_with_kvcache(
                q.unsqueeze(1), # [batch, 1, num_heads, head_dim]
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
            output = output.squeeze(1)      # [batch, num_heads, head_dim]
        else:
            raise NotImplementedError("Decode without FlashAttention not supported")

        return output


    def _pytorch_attention(self, q, k, v, context) -> torch.Tensor:
        """PyTorch 原生 Attention（Prefill fallback）
        
        警告：这个实现不支持 paged attention，仅用于测试。
        """
        num_tokens = q.shape[0]

        # GQA： 扩展KV heads
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # 转换形状：[num_tokens, num_heads, head_dim] -> [1, num_heads, num_tokens, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # 计算attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        # 因果编码
        causal_mask = torch.triu(
            torch.full((num_tokens, num_tokens), float("-inf"), device=q.device),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)

        # 转换回[num_tokens, num_heads, head_dim]
        output = output.squeeze(0).transpose(0, 1)

        return output
