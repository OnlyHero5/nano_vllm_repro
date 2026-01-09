"""
用户请求 "Hello, how are you?"
        ↓
   Tokenize → [15496, 11, 703, 527, 499, 30]
        ↓
┌─────────────────────────────────────────────────────────────────┐
│  Sequence 创建                                                   │
│  seq_id: 0                                                      │
│  status: WAITING                                                │
│  token_ids: [15496, 11, 703, 527, 499, 30]                     │
│  num_prompt_tokens: 6                                           │
│  block_table: []  ← 还未分配显存                                │
└─────────────────────────────────────────────────────────────────┘
        ↓ scheduler.schedule() 调度
┌─────────────────────────────────────────────────────────────────┐
│  Sequence 运行                                                   │
│  status: RUNNING                                                │
│  block_table: [0]  ← BlockManager 分配了物理块 0                │
└─────────────────────────────────────────────────────────────────┘
        ↓ Prefill: 处理完整 prompt
        ↓ Decode: 逐个生成 token
┌─────────────────────────────────────────────────────────────────┐
│  生成过程 (假设生成了 "I am fine")                              │
│  token_ids: [15496, 11, 703, 527, 499, 30, 40, 716, 7024]      │
│  num_tokens: 9                                                  │
│  num_completion_tokens: 3                                       │
└─────────────────────────────────────────────────────────────────┘
        ↓ 遇到 EOS 或达到 max_tokens
┌─────────────────────────────────────────────────────────────────┐
│  Sequence 完成                                                   │
│  status: FINISHED                                               │
│  block_table: [0]  ← 待释放                                     │
└─────────────────────────────────────────────────────────────────┘
"""



from copy import copy
from enum import Enum, auto
from itertools import count

from sampling_params import SamplingParams

class SequenceStatus(Enum):
    """
    序列状态枚举

    状态包括：waiting -> running -> finished
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()



class Sequence:
    """
    序列类 

    每个用户请求封装成Sequence对象，
    负责管理token列表、状态、以及 KV Cache的块映射

    关键属性：
    - token_ids: 完整的token序列
    - block-table: 物理块ID列表 （PagedAttention 核心）
    - status: 当前状态
    """
    block_size = 256 # KV Cache块的大小
    counter = count() # 全局计数器，生成唯一 seq_id

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        """
        初始化序列

        Args:
            token_ids: prompt 的 token ID列表
            sampling_params: 采样参数
        """

        # ===基本属性=== 
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        # ===Token管理=== 
        self.token_ids = copy(token_ids) # 实际深拷贝, list中int为不可变，非嵌套结构
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids) # 可变的目前的总token数
        self.num_prompt_tokens = len(token_ids) # 不可变的prompt token数

        # ===PagedAttention=== 
        self.num_cached_tokens = 0 # 已缓存的token数 (Prefix Caching)
        self.block_table = [] # 物理块ID列表

        #===采样参数 ===
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前的token总数"""
        return self.num_tokens
    
    def __getitem__(self, key):
        return self.token_ids[key]

    # ===状态查询=== 

    @property
    def is_finished(self):
        """检查是否完成"""
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        """已经生成的token = 总token - prompt数"""
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def prompt_token_ids(self):
        """获取prompt部分的token"""
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self):
        """获取生成部分的token"""
        return self.token_ids[self.num_prompt_tokens:]
    
    # ===Block计算相关=== 

    @property
    def num_cached_blocks(self):
        """已经缓存的完整块数"""
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self):
        """当前需要的总块数 = ceil(num_tokens / block_size)"""
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        """最后一个块中的token数"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    @property
    def block(self, i: int):
        """获取第i个块的token列表"""
        assert 0 <= i < self.num_blocks, f"块索引越界：{i}"
        return self.token_ids[i * self.num_blocks: (i+1) * self.block_size]
    
    # ===核心操作===
    def append_token(self, token_id: int):
        """
        追加一个新生成的token

        每次Decode阶段生成一个token后调用
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    # ===序列化支持（优化多线程通信）===
    def __getstate__(self):
        """
        自定义pickle 序列化
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
            )
    
    def __setstate__(self, state):
        """反序列化"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
    
    