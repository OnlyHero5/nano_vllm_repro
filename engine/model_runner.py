"""模型执行器

ModelRunner 是连接 Scheduler 和 Model 的桥梁。

职责：
1. 加载模型和 tokenizer
2. 分配 KV Cache 显存
3. 准备模型输入（构建 Context）
4. 执行模型前向传播
5. 调用 Sampler 生成 token

数据流：
Scheduler.schedule() -> sequences
    ↓
ModelRunner.prepare_xxx() -> 构建 input_ids, positions, Context
    ↓
Model.forward() -> logits
    ↓
Sampler.forward() -> next_tokens
"""
import torch
from torch import nn
from transformers import AutoTokenizer
from typing import Optional

from config import Config
from engine.sequence import Sequence
from utils.context import Context, set_context, get_context
from utils.loader import load_model
from layers.sampler import Sampler

class ModelRunner:
    """模型执行器
    
    管理模型的加载、KV Cache 分配和推理执行
    """
    def __init__(
            self,
            config: Config):
        """
        Args: 
            config: 全局配置
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )

        # 加载模型
        self.model = self._load_model()

        # 采样器
        self.sampler = Sampler()

        # KV cache
        self.kv_cache: Optional[list[torch.Tensor]] = None

        # 模型配置缓存
        self.num_layers = self.model.config.num_hidden_layers
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = getattr(
            self.model.config,
            "head_dim",
            self.model.config.hidden_size // self.model.config.num_attention_heads
        )

        self.block_size = Sequence.block_size

    def _load_model(self) -> nn.Module:
        """加载模型并移至GPU"""
        from models.qwen3 import Qwen3ForCausalLM
        
        print(f"[ModelRunner] 加载模型：{self.config.model_path}")

        # 创建模型结构
        model = Qwen3ForCausalLM.from_pretrained(self.config.model_path)

        # 加载权重
        load_model(model, self.config.model_path)

        # 迁移至GPU 并设置为评估模式
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()

        print(f"[ModelRunner] 模型加载完成，设备：{self.device}")
        return model
    
    def allocate_kv_cache(self, num_blocks: int):
        """预分配 KV Cache 显存
        
        KV Cache 结构：每层一个 tensor
        Shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
        - 2: K 和 V
        - num_blocks: 总块数
        - block_size: 每块的 token 数（如 16）
        - num_kv_heads: KV 头数（GQA）
        - head_dim: 每个头的维度
        
        显存计算：
        每层: 2 * num_blocks * block_size * num_kv_heads * head_dim * 2 bytes (fp16)
        总计: num_layers * 上述值
        
        Args:
            num_blocks: 要分配的块数
        """
        
        # 计算显存需求
        bytes_per_block = (
            2 * 
            self.block_size *
            self.num_kv_heads *
            self.head_dim * 
            2
        )
        total_bytes = self.num_layers * num_blocks * bytes_per_block
        print(f"[ModelRunner] KV Cache 显存需求：{total_bytes / 1024**3:.2f} GB")

        # 分配
        self.kv_cache = []
        for _ in range(self.num_layers):
            # [2, num_blocks, block_size, num_kv_heads, head_dim]
            cache = torch.zeros(
                2,
                num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=torch.float16,
                device=self.device
            )
            self.kv_cache.append(cache)

        print(f"[ModelRunner] KV Cache 分配完成：{num_blocks} 块 × {self.num_layers} 层")

    def get_num_free_gpu_blocks(self) -> int:
        """计算可用的 KV Cache 块数
        
        基于 GPU 显存和配置的 gpu_memory_utilization 计算。
        """
        if not torch.cuda.is_available():
            return 100 # CPU 默认值
        
        # 获取GPU显存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory

        # 应用memory_utilization 系数
        available_memory = free_memory * self.config.gpu_memory_utilization

        # 计算每块的显存需求
        bytes_per_block_per_layer = (
            2 *
            self.block_size *
            self.num_kv_heads *
            self.head_dim *
            2
        )
        bytes_per_block = bytes_per_block_per_layer * self.num_layers

        num_blocks = int(available_memory // bytes_per_block)

        print(f"[ModelRunner] GPU显存：{total_memory / 1024**3: .1f} GB 总计,"
              f"{free_memory / 1024**3: .1f} GB空闲")
        print(f"[ModelRunner] 可分配KV Cache 块数：{num_blocks}")

        return num_blocks
    
    def prepare_prefill(
            self,
            sequences: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """准备 Prefill 阶段的输入
        
        Prefill: 处理完整的 prompt，一次性计算所有 token 的 attention。
        
        Args:
            sequences: 要处理的序列列表
        
        Returns:
            (input_ids, positions): 模型输入
        
        同时设置全局 Context 供 Attention 层使用。
        """
        # 收集所有token
        all_token_ids = []
        all_positions = []
        cu_seqlens = [0] # 累积序列长度, flash-attn
        slot_mapping = [] # 每个Token在KV Cache中的位置

        for seq in sequences:
            token_ids = seq.token_ids
            seq_len = len(token_ids)

            # 收集token_ids和位置
            all_token_ids.extend(token_ids)
            all_positions.extend(range(seq_len))

            # 累积序列长度
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

            # 计算slot mapping
            # slot = block_id * block_size + offset_in_block
            for i in range(seq_len):
                block_idx = i // self.block_size
                offset = i % self.block_size

                if block_idx < len(seq.block_table):
                    block_id = seq.block_table[block_idx]
                    slot = block_id * self.block_size + offset
                    slot_mapping.append(slot)
                else:
                    # 防御性编程
                    slot_mapping.append(0)
        
        # 转换为Tensor
        input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=self.device)

        # 计算最大序列长度
        max_seqlen = max(len(seq.token_ids) for seq in sequences)

        # 设置全局 Context
        context = Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=torch.tensor(slot_mapping,dtype=torch.long, device=self.device),
            # prefill不需要以下字段
            context_lens=None,
            block_tables=None,
            max_context_len=None,
            max_num_blocks=None,

            kv_cache=self.kv_cache
        )
        set_context(context)

        return input_ids, positions
    
    def prepare_decode(
            self, 
            sequences: list[Sequence]
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """准备 Decode 阶段的输入
        
        Decode: 每个序列只处理最新的 1 个 token。
        
        Args:
            sequences: 要处理的序列列表
        
        Returns:
            (input_ids, positions): 模型输入
        """
        # 每个序列只取最后一个token
        input_ids = []
        positions = []
        context_lens = []   # 每个序列的上下文长度
        block_tables = []   # 每个序列的块表
        slot_mapping = []   # 新token的存储位置

        max_num_blocks = max(len(seq.block_table) for seq in sequences) if sequences else 0

        for seq in sequences:
            # 最后一个token
            input_ids.append(seq.last_token)

            # 位置是当前序列长度 -1 
            positions.append(seq.num_tokens - 1)

            # 上下文长度
            context_lens.append(seq.num_tokens)

            # 块表
            padded_block_table = seq.block_table.copy()
            while len(padded_block_table) < max_num_blocks:
                padded_block_table.append(0)
            block_tables.append(padded_block_table)

            # 新token的slot
            # 位置 = num_tokens - 1
            pos = seq.num_tokens - 1
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = seq.block_table[block_idx] if block_idx < len(seq.block_table) else 0
            slot = block_id * self.block_size + offset
            slot_mapping.append(slot)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(positions, dtype=torch.long, device=self.device)

        # 设置全局context
        context = Context(
            is_prefill=False,
            # Decode阶段不需要 cu_seqlens
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),
            max_context_len=max(context_lens) if context_lens else 0,
            max_num_blocks=max_num_blocks,
            kv_cache=self.kv_cache
        )
        set_context(context)

        return input_ids, positions



    @torch.inference_mode()
    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool
    ) -> list[int]:
        """执行模型推理
        
        Args:
            sequences: 要处理的序列
            is_prefill: 是否是 Prefill 阶段
        
        Returns:
            每个序列的下一个 token ID
        """
        if not sequences:
            return []
        
        if is_prefill:
            input_ids, positions = self.prepare_prefill(sequences)
        else:
            input_ids, positions = self.prepare_decode(sequences)
        
        # 前向传播
        logits = self.model(input_ids, positions)

        # 只取每个序列最后一个token的logits
        if is_prefill:
            # Prefill: 需要根据cu_seqlens 取每个序列的最后一个
            context = get_context()
            last_token_indices = context.cu_seqlens_q[1:] - 1
            last_token_indices = last_token_indices.long()
            logits = logits[last_token_indices]
        # Decode： logits已经是每个序列一个 [num_seqs, vocab_size]

        # 采样
        temperatures = torch.tensor(
            [seq.temperature for seq in sequences],
            dtype=torch.float32,
            device=self.device
        )
        next_tokens = self.sampler(logits, temperatures)

        return next_tokens.tolist()


    

    
