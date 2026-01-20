import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
from llm import LLM
from sampling_params import SamplingParams

def main():
    model_path = os.path.join(os.path.dirname(__file__), "models/Qwen3-0.6B")

    print("="*60)
    print("nano vllm Test")
    print("="*60)

    llm = LLM(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    raw_prompts = ["你好，请用300字介绍一下你自己。", "1+1=?"]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": p
                }
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in raw_prompts
    ]

    # 生成
    outputs = llm.generate(
        prompts
    )

    for prompt, output in zip(raw_prompts, outputs):
        print(f"\n [问题] {prompt}")
        print(f"[回答] {output['text']}")
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f" CUDA: {torch.cuda.get_device_name(0)}")
    main()