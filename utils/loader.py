"""权重加载工具

从 HuggingFace safetensors 格式加载权重到自定义模型。

核心挑战：
1. HuggingFace 权重是分离的（q_proj, k_proj, v_proj）
2. 我们的模型是融合的（qkv_proj）
3. 需要正确映射和拼接

解决方案：
1. 模型定义 packed_modules_mapping 指定映射规则
2. 融合层的参数绑定 weight_loader 方法
3. 加载时根据映射调用对应的 weight_loader
"""

import os
from glob import glob
from typing import Optional

import torch
from torch import nn

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("[警告] safetensors 未安装")

from layers.linear import default_weight_loader


def load_model(model: nn.Module, model_path: str):
    """加载模型权重
    
    从 HuggingFace 模型目录加载 safetensors 权重到自定义模型。
    
    Args:
        model: 目标模型（需要有 packed_modules_mapping 属性）
        model_path: HuggingFace 模型目录路径
    
    工作流程：
    1. 获取模型的 packed_modules_mapping
    2. 遍历所有 .safetensors 文件
    3. 对每个权重：
       - 检查是否需要融合加载（在 mapping 中）
       - 是：转换名称，调用 weight_loader(param, weight, shard_id)
       - 否：直接调用 default_weight_loader(param, weight)
    
    权重名称示例：
    HuggingFace:
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
    我们的模型：
        model.layers.0.self_attn.qkv_proj.weight（融合）
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors is required for loading weights")
    
    # 获取融合映射规律
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 统计
    loaded_count = 0
    skipped_count = 0

    # 遍历所有safetensors文件
    safetensor_files = glob(os.path.join(model_path, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    
    print(f"[Loader] 发现 {len(safetensor_files)} 个权重文件")

    for file_path in safetensor_files:
        print(f"[Loader] 加载：{os.path.basename(file_path)}")

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)

                # 检查是否需要融合加载
                is_packed = False
                for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
                    if original_name in weight_name:
                        # 需要融合的层
                        param_name = weight_name.replace(original_name, packed_name)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            # 预防可能的错误
                            print(f"[Loader] 跳过（参数不存在）: {param_name}")
                            skipped_count += 1
                            is_packed = True
                            break
                        
                        # 获取并调用weight_loader
                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is None:
                            raise RuntimeError(
                                f"参数{param_name}没有weight_loader方法，"
                                f"请确保使用 QKVLinear 或 MergedLinear"
                            )

                        weight_loader(param, loaded_weight, shard_id)
                        loaded_count += 1
                        is_packed = True
                        break
                if not is_packed:
                    # 普通权重，直接加载
                    try:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)

                        # 普通 weight_loader 
                        if weight_loader == default_weight_loader:
                            weight_loader(param, loaded_weight)
                        else:
                            # RowLinear 的 weight_loader
                            weight_loader(param, loaded_weight)
                        
                        loaded_count += 1
                    except AttributeError:
                        # 参数不存在
                        skipped_count += 1
    print(f"[Loader] 完成：加载{loaded_count} 个权重，跳过 {skipped_count}个")

def load_model_weights(model: nn.Module, model_path: str):
    """别名"""
    return load_model(model, model_path)