from dataclasses import dataclass

@dataclass
class SamplingParams:
    """
    采样参数配置

    控制文本生成时的随机性和长度

    包括 温度参数、最大生成token 和 是否忽略 EOS token（强制生成到max_tokens）
    """

    temperature: float = 1.0

    max_tokens: int = 4096

    ignore_eos: bool = False

    def __post_init__(self):
        """参数校验"""
        assert self.temperature > 1e-10, "temperature 必须 > 0"
