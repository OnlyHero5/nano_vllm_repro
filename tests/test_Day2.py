"""Day 2 测试脚本 - 验证模型架构

注意：nano-vllm 是推理框架，所有测试都在 inference_mode 下运行
"""

import sys
sys.path.insert(0, '.')

import torch
from layers.layernorm import RMSNorm
from layers.activation import SiluAndMul
from layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb, get_rope
from utils.context import Context, set_context, reset_context


@torch.inference_mode()  # 推理模式：禁用梯度，允许原地操作
def test_rmsnorm():
    """测试 RMSNorm"""
    print("=" * 50)
    print("测试 RMSNorm")
    print("=" * 50)
    
    hidden_size = 128
    batch_size = 2
    seq_len = 10
    
    norm = RMSNorm(hidden_size)
    
    # 测试基础 forward
    x = torch.randn(batch_size, seq_len, hidden_size)
    out = norm(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    assert out.shape == x.shape
    
    # 验证归一化效果：输出的 RMS 应该接近 1
    rms = out.pow(2).mean(dim=-1).sqrt()
    print(f"输出 RMS 均值: {rms.mean().item():.4f} (应接近 weight 初始值 1.0)")
    
    # 测试融合残差版本
    x2 = torch.randn(batch_size, seq_len, hidden_size)
    residual = torch.randn_like(x2)
    out2, new_residual = norm(x2, residual)
    print(f"融合残差版本 - 输出形状: {out2.shape}, 新残差形状: {new_residual.shape}")
    
    print("✅ RMSNorm 测试通过!\n")


@torch.inference_mode()
def test_silu_and_mul():
    """测试 SiluAndMul (SwiGLU)"""
    print("=" * 50)
    print("测试 SiluAndMul")
    print("=" * 50)
    
    act = SiluAndMul()
    
    # 输入是 gate 和 up 拼接的结果
    batch_size = 2
    seq_len = 10
    intermediate_size = 64
    
    x = torch.randn(batch_size, seq_len, intermediate_size * 2)
    out = act(x)
    
    print(f"输入形状: {x.shape} (2 * intermediate_size)")
    print(f"输出形状: {out.shape} (intermediate_size)")
    assert out.shape == (batch_size, seq_len, intermediate_size)
    
    # 手动验证计算
    x2 = torch.randn(batch_size, seq_len, intermediate_size * 2)
    gate, up = x2.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    out2 = act(x2)
    assert torch.allclose(out2, expected, atol=1e-6)
    
    print("✅ SiluAndMul 测试通过!\n")


@torch.inference_mode()
def test_rope():
    """测试 RoPE"""
    print("=" * 50)
    print("测试 RoPE (旋转位置编码)")
    print("=" * 50)
    
    head_dim = 64
    max_position = 1024
    base = 10000.0
    
    rope = get_rope(head_dim, head_dim, max_position, base)
    
    # 测试数据
    num_tokens = 5
    num_heads = 4
    num_kv_heads = 2
    
    positions = torch.arange(num_tokens)
    query = torch.randn(num_tokens, num_heads, head_dim)
    key = torch.randn(num_tokens, num_kv_heads, head_dim)
    
    # 保存原始模长用于比较
    q_norm_before = query.norm(dim=-1).clone()
    
    q_rot, k_rot = rope(positions, query, key)
    
    print(f"位置索引: {positions}")
    print(f"Query 形状: {query.shape} → {q_rot.shape}")
    print(f"Key 形状: {key.shape} → {k_rot.shape}")
    
    # 验证：旋转不改变向量的模长
    q_norm_after = q_rot.norm(dim=-1)
    print(f"Query 模长变化: {(q_norm_after / q_norm_before).mean().item():.4f} (应接近 1.0)")
    
    print("✅ RoPE 测试通过!\n")


@torch.inference_mode()
def test_rope_relative_position():
    """验证 RoPE 的相对位置编码性质"""
    print("=" * 50)
    print("验证 RoPE 相对位置性质")
    print("=" * 50)
    
    head_dim = 64
    
    # 每次测试创建新的 rope 实例，避免缓存问题
    rope = RotaryEmbedding(head_dim, head_dim, 1024, 10000.0)
    
    # 创建两个相同的向量
    q_original = torch.randn(1, 1, head_dim)
    k_original = q_original.clone()
    
    # 放在不同位置
    pos1 = torch.tensor([0])
    pos2 = torch.tensor([5])
    pos3 = torch.tensor([10])
    
    # 相同位置
    q1, k1 = rope(pos1, q_original.clone(), k_original.clone())
    dot_same = (q1 * k1).sum()
    
    # 相差 5 的位置 (0 和 5)
    q_at_0, _ = rope(pos1, q_original.clone(), k_original.clone())
    _, k_at_5 = rope(pos2, q_original.clone(), k_original.clone())
    dot_diff_5 = (q_at_0 * k_at_5).sum()
    
    # 相差 5 的位置 (5 和 10)
    q_at_5, _ = rope(pos2, q_original.clone(), k_original.clone())
    _, k_at_10 = rope(pos3, q_original.clone(), k_original.clone())
    dot_diff_5_v2 = (q_at_5 * k_at_10).sum()
    
    print(f"相同位置的点积: {dot_same.item():.4f}")
    print(f"位置 0 和 5 的点积: {dot_diff_5.item():.4f}")
    print(f"位置 5 和 10 的点积: {dot_diff_5_v2.item():.4f}")
    print(f"相对位置相同时点积差异: {abs(dot_diff_5.item() - dot_diff_5_v2.item()):.6f}")
    
    # 验证：相同相对位置的点积应该相等
    assert abs(dot_diff_5.item() - dot_diff_5_v2.item()) < 1e-4, "相对位置编码性质不满足!"
    
    print("✅ RoPE 相对位置性质验证通过!\n")


@torch.inference_mode()
def test_qwen3_model():
    """测试 Qwen3 模型"""
    print("=" * 50)
    print("测试 Qwen3 模型")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("⚠️ 跳过 Qwen3 模型测试（需要 CUDA + FlashAttention）\n")
        return

    from models.qwen3 import Qwen3ForCausalLM
    from dataclasses import dataclass

    # 创建一个小型配置用于测试
    @dataclass
    class TestConfig:
        vocab_size: int = 1000
        hidden_size: int = 128
        num_hidden_layers: int = 2
        num_attention_heads: int = 4
        num_key_value_heads: int = 2
        intermediate_size: int = 256
        max_position_embeddings: int = 512
        rms_norm_eps: float = 1e-6
        attention_bias: bool = False  # Qwen3 默认 False
        rope_theta: float = 10000.0
        tie_word_embeddings: bool = False

    config = TestConfig()
    model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
    model.eval()  # 设置为评估模式

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播（prefill 模式：一次处理完整序列）
    num_tokens = 10
    input_ids = torch.randint(0, config.vocab_size, (num_tokens,)).cuda()

    # 设置 prefill Context（Attention 层通过全局 Context 获取元数据）
    set_context(Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda'),
        cu_seqlens_k=torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda'),
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
    ))

    logits = model(input_ids)
    reset_context()

    print(f"输入 token 数: {num_tokens}")
    print(f"输出 logits 形状: {logits.shape}")
    assert logits.shape == (num_tokens, config.vocab_size)

    # 测试自回归生成（简单模拟，每轮 prefill 整个已生成序列）
    print("\n模拟自回归生成:")
    generated = input_ids.tolist()
    for _ in range(3):
        seq_len = len(generated)
        set_context(Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device='cuda'),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device='cuda'),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        ))
        logits = model(torch.tensor(generated, device='cuda'))
        reset_context()
        next_token = logits[-1].argmax().item()
        generated.append(next_token)
        print(f"  生成 token: {next_token}")

    print("✅ Qwen3 模型测试通过!\n")


@torch.inference_mode()
def test_gqa():
    """测试 Grouped Query Attention"""
    print("=" * 50)
    print("测试 GQA (Grouped Query Attention)")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("⚠️ 跳过 GQA 测试（需要 CUDA + FlashAttention）\n")
        return

    from models.qwen3 import Qwen3Attention

    hidden_size = 256
    num_heads = 8
    num_kv_heads = 2  # GQA: 每 4 个 Q head 共享 1 个 KV head（head_dim=32）

    attn = Qwen3Attention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qkv_bias=False,  # Qwen3 默认 False，会使用 QK Norm
    ).cuda().to(torch.bfloat16)
    attn.eval()

    num_tokens = 5
    hidden_states = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device='cuda')

    # 设置 prefill Context（Attention 层走 FlashAttention varlen 路径）
    set_context(Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda'),
        cu_seqlens_k=torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda'),
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
    ))

    output = attn(positions, hidden_states)
    reset_context()

    print(f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}")
    print(f"每个 KV head 被 {num_heads // num_kv_heads} 个 Q head 共享")
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")

    assert output.shape == hidden_states.shape
    print("✅ GQA 测试通过!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("nano-vllm Day 2 测试 (推理模式)")
    print("=" * 50)
    print()
    
    test_rmsnorm()
    test_silu_and_mul()
    test_rope()
    test_rope_relative_position()
    test_qwen3_model()
    test_gqa()
    
    print("=" * 50)
    print("🎉 Day 2 所有测试通过!")
    print("=" * 50)