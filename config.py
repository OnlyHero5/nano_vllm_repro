import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    """
    nano-vllm 核心配置类

    用于管理模型路径、推理参数、显存配置等全局设置。
    """

    # 模型路径
    model_path: str

    # 连续批处理 相关参数
    max_num_batched_tokens: int = 16384     #单批次最大token数
    max_num_seqs: int = 512     # 最大并发序列数
    max_model_len: int = 4096   # 最大上下文长度

    # 显存管理
    gpu_memory_utilization: float = 0.7

    # 并行配置
    tensor_parallel_size: int = 1   #张量并行数（1为单卡）

    # 调试选择
    enforce_eager: bool = False     # True=禁用CUDA Graph

    # 运行时自动填充的属性
    hf_config: AutoConfig | None = None
    eos: int = -1

    # PagedAttention 参数
    kvcache_block_size: int = 256   # KV cache 块大小
    num_kvcache_blocks: int = -1    # KV cache 块数量

    def __post_init__(self):
        """
        dataclass 初始化后自动调用，用于参数校验和自动配置
        """

        # 1. 校验模型路径
        assert os.path.isdir(self.model_path), f"模型路径不存在：{self.model_path}"

        # 2. 块大小必须是256的倍数 （FlashAttention优化要求）
        assert self.kvcache_block_size % 256 == 0, f"kvcache_block_size 必须是 256 的倍数"

        # 3. 张量并行数范围检查
        assert 1 <= self.tensor_parallel_size <= 8 , "张量并行必须在1-8之间 单机"

        # 4. 自动加载huggingface 模型配置
        self.hf_config = AutoConfig.from_pretrained(self.model_path)

        # 5. 上下文长度 配置文件和模型支持的最小值
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        # 6. 单批次token数必须 >= 最大上下文长度（确保能处理最长序列）
        assert self.max_num_batched_tokens >= self.max_model_len, "max_num_batched_tokens 必须 >= max_model_len"

        
    @property
    def model(self) -> str:
        """别名"""
        return self.model_path

