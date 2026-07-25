"""验证基础数据结构"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from sampling_params import SamplingParams

from engine.sequence import Sequence, SequenceStatus
from utils.context import Context, get_context, set_context, reset_context
import torch



def test_sampling_params():
    """测试 SamplingParams"""
    print("=" * 50)
    print("测试 SamplingParams")
    print("=" * 50)
    
    # 默认参数
    sp = SamplingParams()
    print(f"默认参数: temperature={sp.temperature}, max_tokens={sp.max_tokens}")
    assert sp.temperature == 1.0
    assert sp.max_tokens == 4096
    assert sp.ignore_eos == False
    
    # 自定义参数
    sp2 = SamplingParams(temperature=0.7, max_tokens=128, ignore_eos=True)
    print(f"自定义参数: temperature={sp2.temperature}, max_tokens={sp2.max_tokens}")
    
    # 测试参数校验
    try:
        SamplingParams(temperature=0)  # 应该失败
        print("❌ 应该抛出异常但没有")
    except AssertionError as e:
        print(f"✅ 正确拒绝 temperature=0: {e}")
    
    print("✅ SamplingParams 测试通过!\n")


def test_sequence():
    """测试 Sequence"""
    print("=" * 50)
    print("测试 Sequence")
    print("=" * 50)
    
    # 模拟 prompt tokens
    prompt_tokens = [15496, 11, 703, 527, 499, 30]  # "Hello, how are you?"
    
    # 创建序列
    seq = Sequence(prompt_tokens, SamplingParams(temperature=0.8, max_tokens=100))
    
    print(f"seq_id: {seq.seq_id}")
    print(f"status: {seq.status}")
    print(f"num_tokens: {seq.num_tokens}")
    print(f"num_prompt_tokens: {seq.num_prompt_tokens}")
    print(f"num_completion_tokens: {seq.num_completion_tokens}")
    print(f"temperature: {seq.temperature}")
    
    # 验证初始状态
    assert seq.status == SequenceStatus.WAITING
    assert len(seq) == 6
    assert seq.num_completion_tokens == 0
    assert seq.is_finished == False
    
    # 测试 block 计算
    print(f"\nBlock 相关属性:")
    print(f"  block_size: {seq.block_size}")
    print(f"  num_blocks: {seq.num_blocks}")  # ceil(6/256) = 1
    print(f"  last_block_num_tokens: {seq.last_block_num_tokens}")
    
    # 模拟生成过程
    print(f"\n模拟 Decode 过程:")
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [0]  # 假设分配了物理块 0
    
    # 生成 3 个 token
    generated_tokens = [40, 716, 7024]  # "I", "am", "fine"
    for token in generated_tokens:
        seq.append_token(token)
        print(f"  生成 token {token}, 当前长度: {len(seq)}")
    
    assert seq.num_tokens == 9
    assert seq.num_completion_tokens == 3
    assert seq.last_token == 7024
    assert seq.completion_token_ids == generated_tokens
    
    # 测试完成状态
    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished == True
    
    print(f"\n最终状态:")
    print(f"  status: {seq.status}")
    print(f"  token_ids: {seq.token_ids}")
    print(f"  prompt_token_ids: {seq.prompt_token_ids}")
    print(f"  completion_token_ids: {seq.completion_token_ids}")
    
    print("✅ Sequence 测试通过!\n")


def test_context():
    """测试 Context"""
    print("=" * 50)
    print("测试 Context")
    print("=" * 50)
    
    # 初始状态
    ctx = get_context()
    print(f"初始状态: is_prefill={ctx.is_prefill}")
    assert ctx.is_prefill == False
    
    # 模拟 Prefill 阶段设置
    set_context(Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 4, 6, 11], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 4, 6, 11], dtype=torch.int32),
        max_seqlen_q=5,
        max_seqlen_k=5,
        slot_mapping=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ))
    
    ctx = get_context()
    print(f"Prefill 阶段:")
    print(f"  is_prefill: {ctx.is_prefill}")
    print(f"  cu_seqlens_q: {ctx.cu_seqlens_q}")
    print(f"  max_seqlen_q: {ctx.max_seqlen_q}")
    assert ctx.is_prefill == True
    
    # 模拟 Decode 阶段设置
    set_context(Context(
        is_prefill=False,
        context_lens=torch.tensor([10, 8, 15]),
        block_tables=torch.tensor([[0, 1], [2, 3], [4, 5]])
    ))
    
    ctx = get_context()
    print(f"\nDecode 阶段:")
    print(f"  is_prefill: {ctx.is_prefill}")
    print(f"  context_lens: {ctx.context_lens}")
    print(f"  block_tables shape: {ctx.block_tables.shape}")
    assert ctx.is_prefill == False
    
    # 重置
    reset_context()
    ctx = get_context()
    assert ctx.is_prefill == False
    assert ctx.cu_seqlens_q is None
    
    print("✅ Context 测试通过!\n")


def test_config():
    """测试 Config（需要有效的模型路径）"""
    print("=" * 50)
    print("测试 Config")
    print("=" * 50)
    
    config = Config(model_path="models/Qwen3-0.6B")
    print(f"模型配置: {config.hf_config}")
    print(f"max_model_len: {config.max_model_len}")
    print(f"kvcache_block_size: {config.kvcache_block_size}")
    


if __name__ == "__main__":
    test_sampling_params()
    test_sequence()
    test_context()
    test_config()
    
    print("=" * 50)
    print("🎉 Day 1 所有测试通过!")
    print("=" * 50)