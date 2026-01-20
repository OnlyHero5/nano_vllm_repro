"""LLM 推理引擎

串联 Scheduler + ModelRunner，实现完整推理循环。
"""
import atexit
from time import perf_counter
from typing import Union

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from config import Config
from sampling_params import SamplingParams
from engine.sequence import Sequence
from engine.block_manager import BlockManager
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner

class LLMEngine:
    """LLM推理引擎"""

    def __init__(self, model:str, **kwargs):
        """
        Args:
            model: 模型路径
            **kwargs: 配置参数
        """
        # 创建配置
        self.config = Config(model_path=model, **kwargs)

        # 加载Tokenizer
        print(f"[LLMEngine] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        self.config.eos = self.tokenizer.eos_token_id
        print(f"[LLMEngine] EOS token ID: {self.config.eos}")

        # 初始化 ModelRunner
        print(f"[LLMEngine] 初始化 ModelRunner...")
        self.model_runner = ModelRunner(self.config)

        # 计算KV Cache 块数
        num_blocks = self.model_runner.get_num_free_gpu_blocks()
        num_blocks = max(1, int(num_blocks * 0.95))

        # 分配KV Cache
        self.model_runner.allocate_kv_cache(num_blocks)
        
        # 创建 BlockManager 
        block_size = Sequence.block_size
        self.block_manager = BlockManager(num_blocks, block_size)

        # 创建 Scheduler 
        self.scheduler= Scheduler(self.config, self.block_manager)

        atexit.register(self._cleanup)

        print(f"[LLMEngine] 初始化完成！")
        print(f"[LLMEngine] - KV Cache: {num_blocks} 块")
        print(f"[LLMEngine] - Block Size: {block_size} tokens")
    
    def _cleanup(self):
        """清理资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    



    def add_request(
            self,
            prompt: Union[str, list[int]],
            sampling_params: SamplingParams = None
    ):
        """添加推理请求"""
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Tokenize
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            token_ids = list(prompt)
        
        # 创建序列并添加
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add_sequence(seq)

    def is_finished(self) -> bool:
        """检查是否完成"""
        return self.scheduler.is_finished()


    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """执行单步推理
        
        Returns:
            (outputs, num_tokens)
        """
        # 调度
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return [], 0
        
        # 执行推理
        token_ids = self.model_runner.run(seqs, is_prefill)

        # 后处理
        finished_seqs = self.scheduler.postprocess(seqs, token_ids)

        # 收集输出
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in finished_seqs
        ]

        # 计算token数
        if is_prefill:
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            num_tokens = -len(seqs)
        
        return outputs, num_tokens
    
    def generate(
            self,
            prompts: Union[list[str], list[list[int]]],
            sampling_params: Union[SamplingParams, list[SamplingParams]] = None,
            use_tqdm: bool = True
    ) -> list[dict]:
        """批量生成"""
        if sampling_params is None:
            sampling_params = SamplingParams()

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 添加请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # 进度条
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        outputs = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0

        # 生成循环
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - t

            if pbar and elapsed > 0:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                elif num_tokens < 0: # decode阶段
                    decode_throughput = -num_tokens / elapsed
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)} tok/s",
                        "Decode": f"{int(decode_throughput)} tok/s"
                    }
                )
            
            for seq_id, token_id in output:
                outputs[seq_id] = token_id
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
        
        # 排序并解码
        sorted_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        results = []
        for token_ids in sorted_outputs:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append(
                {
                    "text": text,
                    "token_ids": token_ids
                }
            )
        return results