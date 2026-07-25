# Day 7 — 进阶优化：CUDA Graph 与 Tensor Parallel 入门

## 本篇定位

到 Day6 为止，你已经拥有了一个**完整的、可运行的单卡推理引擎**。本篇是「进阶加餐」，讲解两个实际 vLLM 中的核心优化技术：

1. **CUDA Graph**：把 Decode 阶段的 GPU 操作「录制成一张图」，消除 CPU launch overhead
2. **Tensor Parallel**：把模型权重切分到多张 GPU，突破单卡显存限制

这两个特性在原版 nano-vLLM（GitHub 14K stars）中都有实现。本篇的代码是**教学版本**——只保留核心思想，不引入分布式进程池等复杂基础设施。

---

## 1. CUDA Graph — 为什么 Decode 适合画图？

### 1.1 背景：CPU Launch Overhead

GPU 执行计算的流程通常是：

```
CPU 说: "来，算个矩阵乘法"     ← kernel launch（~5-20μs）
GPU 做: 矩阵乘法               ← 实际计算（可能很短，尤其是 decode 时）
CPU 说: "来，算个 attention"
GPU 做: attention
CPU 说: "来，算个 MLP"
GPU 做: MLP
...（一次 decode 要 launch 几十个 kernel）
```

对于 decode 阶段，每次只算 **1 个新 token**，每个 kernel 的计算量非常小。CPU 发指令的开销（~5-20μs per launch）反而成为瓶颈。

### 1.2 CUDA Graph 的解决思路

> **"录下来，以后直接重放。"**

```text
录制阶段 (Capture):
  把 decode 的所有 GPU 操作「录」进一张图
  图里保存的是 GPU 指令序列，不是数据

重放阶段 (Replay):
  把新的输入数据拷进图的 buffer
  一次 API 调用 → GPU 执行整张图（几十个 kernel 一气呵成）
```

### 1.3 为什么只在 Decode 用？

| 阶段 | 输入形状 | 适合 CUDA Graph？ |
|------|---------|------------------|
| Prefill | 每条序列的 prompt 长度不同，batch 组合每次变化 | ❌ 形状不固定 |
| Decode | 每条序列只输入 1 个 token，batch_size 固定时形状完全一致 | ✅ 形状固定 |

### 1.4 为什么 Sampler 不放进 Graph？

采样（softmax → Gumbel-Max）有随机性，且每次 batch_size 可能不同。放进 graph 会让重放失去灵活性。

---

## 2. CUDA Graph 教学版实现

### 2.1 前提条件

这段代码依赖 Day4 和 Day5 的修改：
- `Qwen3ForCausalLM.forward()` 返回 hidden states
- `Qwen3ForCausalLM.compute_logits()` 单独可用
- `ModelRunner.run_model()` 已经拆出来

### 2.2 修改 `engine/model_runner.py`（添加 CUDA Graph 支持）

在 `ModelRunner.__init__()` 末尾添加：

```python
# engine/model_runner.py —— 在 ModelRunner.__init__() 末尾添加

# CUDA Graph 相关状态
# key = batch_size, value = DecodeGraphRunner
self.decode_graphs: dict[int, "DecodeGraphRunner"] = {}

# 基础版：只在单卡 CUDA 且未禁用 eager 时启用
self.use_cuda_graph = (
    torch.cuda.is_available()
    and not self.config.enforce_eager
    and self.config.tensor_parallel_size == 1
)
```

在 `allocate_kv_cache()` 末尾添加 graph capture 调用：

```python
# engine/model_runner.py —— 在 allocate_kv_cache() 方法末尾添加

if self.use_cuda_graph:
    self.capture_decode_graphs()
```

### 2.3 新增 `DecodeGraphRunner` 数据类

在 `class ModelRunner` 之前添加：

```python
# engine/model_runner.py —— 在 class ModelRunner 前添加

from dataclasses import dataclass


@dataclass
class DecodeGraphRunner:
    """
    一个 batch_size 对应一套静态 CUDA Graph 资源。

    为什么需要静态 buffer？
    CUDA Graph replay 时不能换张量对象（地址必须在录制时就确定），
    所以 input_ids、positions、slot_mapping 等都要提前分配好。
    每次 replay 前，只把新 batch 的数据 copy 到这些静态张量里。

    属性:
        batch_size: 这个图对应多大的 batch
        max_num_blocks: 最大 block_table 列数
        input_ids: 静态输入 token ID tensor [batch_size]
        positions: 静态位置 tensor [batch_size]
        slot_mapping: 静态 slot 映射 tensor [batch_size]
        context_lens: 静态上下文长度 tensor [batch_size]
        block_tables: 静态块表 tensor [batch_size, max_num_blocks]
        hidden_states: 图的输出（模型主干产出的 hidden states）
        graph: 录制好的 torch.cuda.CUDAGraph 对象
    """
    batch_size: int
    max_num_blocks: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    hidden_states: torch.Tensor
    graph: torch.cuda.CUDAGraph
```

### 2.4 新增 `capture_decode_graphs()` 方法

```python
# engine/model_runner.py —— 在 ModelRunner 类中添加 capture_decode_graphs()

@torch.inference_mode()
def capture_decode_graphs(self) -> None:
    """
    录制一组常用 batch_size 的 CUDA Graph。

    基础版只录小档位: 1, 2, 4, 8, 16
    batch_size 没命中时就回退到 eager 模式。

    为什么在 KV Cache 分配之后才调用？
    因为 graph 录制时 Attention 会读写真实的 KV Cache tensor。
    KV Cache 没分配好时，graph 里引用的对象不稳定。
    """
    capture_batch_sizes = [1, 2, 4, 8, 16]
    # 每个序列最多需要多少个 block
    max_num_blocks = max(1, self.config.max_model_len // self.block_size + 1)

    for batch_size in capture_batch_sizes:
        # --- 创建静态 buffer ---
        # 这些张量的地址在录制后不能变，所以这里创建后一直复用
        input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        context_lens = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        block_tables = torch.zeros(
            batch_size, max_num_blocks,
            dtype=torch.int32, device=self.device,
        )

        # --- 设置一个假的 decode Context（Attention 层需要这些元数据） ---
        set_context(Context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            max_context_len=1,
            max_num_blocks=max_num_blocks,
            kv_cache=self.kv_cache,
        ))

        # --- warmup: 先跑一次，避免第一次执行的初始化开销进入 graph ---
        hidden_states = self.model(input_ids, positions)
        torch.cuda.synchronize()  # 确保 GPU 操作全部完成

        # --- 正式录制 ---
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # graph 里的操作：模型主干前向
            # 注意：这里只录 model.forward()，不录 compute_logits() 和 sampler
            hidden_states = self.model(input_ids, positions)

        # --- 保存这个 batch_size 的 graph ---
        self.decode_graphs[batch_size] = DecodeGraphRunner(
            batch_size=batch_size,
            max_num_blocks=max_num_blocks,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            hidden_states=hidden_states,
            graph=graph,
        )

        # --- 清理上下文（避免污染下一轮） ---
        reset_context()

    print(f"[ModelRunner] CUDA Graph 录制完成：{list(self.decode_graphs.keys())} 个 batch_size 档位")
```

### 2.5 更新 `run_model()` — 自动选择 Eager 或 Graph

```python
# engine/model_runner.py —— 更新 run_model() 支持 CUDA Graph

@torch.inference_mode()
def run_model(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    is_prefill: bool,
) -> torch.Tensor:
    """
    graph-aware 的模型执行入口。

    规则（按优先级）:
    1. Prefill → 永远走 eager（输入形状不固定）
    2. Graph 未启用 → eager
    3. Decode batch_size 没命中已录制的档位 → eager
    4. 命中 → CUDA Graph replay
    """
    # --- 规则 1 & 2: Prefill 或未启用 graph → eager ---
    if is_prefill or not self.use_cuda_graph:
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)

        if is_prefill:
            context = get_context()
            # Prefill 只取每条序列最后一个位置的 logits
            last_token_indices = context.cu_seqlens_q[1:] - 1
            logits = logits[last_token_indices.long()]
        return logits

    # --- 规则 3: 查找是否命中 ---
    batch_size = input_ids.shape[0]
    runner = self.decode_graphs.get(batch_size)

    if runner is None:
        # 没录制过这个 batch_size → 回退 eager
        hidden_states = self.model(input_ids, positions)
        return self.model.compute_logits(hidden_states)

    # --- 规则 4: CUDA Graph replay ---
    context = get_context()

    # 步骤 A: 把本轮真实数据写入静态 buffer
    #   注意：不能直接赋值（会改变张量地址），必须用 copy_()
    runner.input_ids.copy_(input_ids)
    runner.positions.copy_(positions)

    # slot_mapping 可能比 buffer 短，先清零再拷贝
    runner.slot_mapping.zero_()
    runner.slot_mapping[:batch_size].copy_(context.slot_mapping)

    runner.context_lens.zero_()
    runner.context_lens[:batch_size].copy_(context.context_lens)

    runner.block_tables.zero_()
    runner.block_tables[:batch_size, :context.block_tables.shape[1]].copy_(
        context.block_tables
    )

    # 步骤 B: 用静态 buffer 重建 Context（graph 里的 Attention 会读取这些）
    set_context(Context(
        is_prefill=False,
        slot_mapping=runner.slot_mapping,
        context_lens=runner.context_lens,
        block_tables=runner.block_tables,
        max_context_len=int(runner.context_lens.max().item()),
        max_num_blocks=runner.max_num_blocks,
        kv_cache=self.kv_cache,
    ))

    # 步骤 C: 重放 graph
    runner.graph.replay()

    # 步骤 D: lm_head 投影（在 graph 外面）
    return self.model.compute_logits(runner.hidden_states)
```

### 2.6 修改 `engine/llm_engine.py` — `enforce_eager` 参数透传

在 `LLMEngine.__init__()` 中确保 `enforce_eager` 被正确传递：

```python
# engine/llm_engine.py —— LLMEngine.__init__() 中

# 如果传了 enforce_eager，覆盖配置
if "enforce_eager" in kwargs:
    self.config.enforce_eager = kwargs.pop("enforce_eager")
```

---

## 3. Tensor Parallel — 教学版实现

### 3.1 核心原理

一块 GPU 的显存放不下大模型怎么办？把权重**切分**到多张 GPU 上。

Tensor Parallel 有两种切法：

| 切法 | 切哪个维度 | 什么时候需要 all_reduce | 例子 |
|------|-----------|------------------------|------|
| **Column Parallel** | 输出维（列） | 不需要（每张卡产出的是局部结果） | QKV 投影 |
| **Row Parallel** | 输入维（行） | 需要（每张卡的局部结果要加起来） | O 投影 |

以 QKV 投影为例，一个形状为 `[hidden_size, total_qkv_size]` 的权重：

```text
单卡（全部）:
  GPU 0: [hidden_size, q_size + kv_size + kv_size]  ← 完整 QKV

Column Parallel（按输出维切到 2 张卡）:
  GPU 0: [hidden_size, q_size/2 + kv_size/2 + kv_size/2]  ← 每个 head 的一半
  GPU 1: [hidden_size, q_size/2 + kv_size/2 + kv_size/2]  ← 另一半
```

### 3.2 修改 `layers/linear.py`（添加 TP helper）

```python
# layers/linear.py —— 在文件顶部 import 之后添加 TP helper

import torch.distributed as dist


def divide(numerator: int, denominator: int) -> int:
    """
    TP 中的整除切分。
    TP 维度必须能被 world_size 整除，否则直接报错。
    """
    assert denominator > 0, "denominator 必须 > 0"
    assert numerator % denominator == 0, (
        f"{numerator} 不能被 {denominator} 整除，无法做 TP 切分"
    )
    return numerator // denominator


def get_tp_world_size() -> int:
    """
    获取当前 TP 的 world_size。
    
    如果没有初始化分布式环境（单卡运行），返回 1。
    这个设计让单卡测试不需要任何分布式代码。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tp_rank() -> int:
    """
    获取当前 TP 的 rank。
    单卡运行时返回 0。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
```

### 3.3 新增 TP 版本的 Linear 基类

```python
# layers/linear.py —— 在现有类之前添加

class LinearBase(nn.Module):
    """
    TP Linear 的公共基类。
    
    tp_dim 表示权重按哪个维度切:
    - 0: 按输出维切（Column Parallel）
    - 1: 按输入维切（Row Parallel）
    - None: 不切（单卡版本）
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
```

### 3.4 Column Parallel Linear（QKV 用）

```python
# layers/linear.py

class ColumnParallelLinear(LinearBase):
    """
    按输出维切分的 Linear（Column Parallel）。
    
    原始权重: [global_output_size, input_size]
    每卡存:   [global_output_size / tp_size, input_size]
    
    典型用途: QKV 投影、Gate-Up 投影
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        local_output_size = divide(output_size, get_tp_world_size())
        super().__init__(input_size, local_output_size, bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """当前 rank 只加载自己那一段输出维"""
        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)
```

### 3.5 Row Parallel Linear（O 投影用）

```python
# layers/linear.py

class RowParallelLinear(LinearBase):
    """
    按输入维切分的 Linear（Row Parallel）。
    
    原始权重: [output_size, global_input_size]
    每卡存:   [output_size, global_input_size / tp_size]
    
    典型用途: O 投影、Down 投影
    
    关键：每张卡算完自己的局部输入贡献后，需要 all_reduce 求和。
    """
    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        local_input_size = divide(input_size, get_tp_world_size())
        super().__init__(local_input_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.data.ndim == 1:
            # bias：不按输入维切，直接全量加载
            param.data.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            return

        shard_size = param.data.size(self.tp_dim)
        start = self.tp_rank * shard_size
        shard = loaded_weight.narrow(self.tp_dim, start, shard_size)
        param.data.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # 求和所有 rank 的局部输出
        return y
```

### 3.6 保留公开类名（兼容现有代码）

```python
# layers/linear.py —— 文件末尾保留别名

# 单卡版本：ColumnParallelLinear/RowParallelLinear 在 tp_size=1 时
# 退化为普通 Linear，所以可以直接用它们替代原来的 QKVLinear/RowLinear
QKVLinear = ColumnParallelLinear
MergedLinear = ColumnParallelLinear    # Gate-Up 也是 Column Parallel
RowLinear = RowParallelLinear
```

> **⚠️ 重要兼容性警告**：
>
> 上面的别名替换**不能直接使用**，因为 `loader.py` 的 `packed_modules_mapping` 会这样调用：
> ```python
> weight_loader(param, loaded_weight, shard_id)  # shard_id = "q" / "k" / "v"
> ```
>
> 但 `ColumnParallelLinear.weight_loader` 的签名是 `(self, param, loaded_weight)`，**没有 `shard_id` 参数**。
>
> 原 `QKVLinear._weight_loader` 用 `shard_id` 来决定将 Q/K/V 的哪一段写入融合权重的哪个位置。
> `ColumnParallelLinear.weight_loader` 只按 `tp_rank` 切分，不理解 `shard_id` 语义。
>
> **解决方案**（二选一）：
>
> **方案 A**：给 `ColumnParallelLinear` 增加 `shard_id` 兼容：
> ```python
> class ColumnParallelLinear(LinearBase):
>     def __init__(self, input_size, output_size, bias=False, num_kv_heads=None, head_dim=None):
>         super().__init__(...)
>         self._shard_id_map = {}  # {shard_id: offset}
>
>     def weight_loader(self, param, loaded_weight, shard_id=None):
>         if shard_id is not None:
>             # QKV 融合模式：按 shard_id 定位偏移
>             offset = self._shard_id_map[shard_id]
>             param.data[offset:offset+loaded_weight.shape[0]].copy_(...)
>         else:
>             # 普通 TP 模式：按 tp_rank 切分
>             shard_size = param.data.size(self.tp_dim)
>             start = self.tp_rank * shard_size
>             ...
> ```
>
> **方案 B**：修改 `loader.py`，让它不再传 `shard_id`，而是由 Linear 自己计算偏移。
>
> **对于单卡学习**：建议保留原有的 `QKVLinear`/`MergedLinear`/`RowLinear`，不做替换。
> TP 改造是一个独立的进阶课题，需要同步修改 `loader.py` 和 `qwen3.py` 的 `packed_modules_mapping`。

### 3.7 修改 `engine/model_runner.py` — 添加 TP 初始化

```python
# engine/model_runner.py —— 在 ModelRunner.__init__() 中添加

def setup_tp_runtime(self) -> None:
    """
    初始化教学版 TP 运行时。

    tensor_parallel_size == 1: 直接单卡运行
    tensor_parallel_size > 1: 需要用 torchrun 启动
    """
    self.tp_size = self.config.tensor_parallel_size
    self.rank = 0
    self.local_rank = 0
    self.is_distributed = False

    if self.tp_size == 1:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Tensor Parallel 需要 CUDA 环境")

    import os
    self.rank = int(os.environ["RANK"])
    self.local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == self.tp_size, (
        f"WORLD_SIZE={world_size} 与 tensor_parallel_size={self.tp_size} 不一致"
    )

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.cuda.set_device(self.local_rank)
    self.device = torch.device("cuda", self.local_rank)
    self.is_distributed = True
    print(f"[ModelRunner] TP 初始化: rank={self.rank}/{self.tp_size}, device={self.device}")
```

### 3.8 启动 TP 的命令

```bash
# 单机 2 卡 TP
torchrun --nproc_per_node=2 example.py

# 如果 example.py 不依赖 TP，就写一个简单的测试脚本
torchrun --nproc_per_node=2 - <<'PY'
import torch
import torch.distributed as dist
dist.init_process_group("nccl")
print(f"Rank {dist.get_rank()}/{dist.get_world_size()} 初始化成功")
PY
```

---

## 4. 知识图谱与总结

### 4.1 完整架构知识图谱

```
┌─────────────────────────────────────────────────────────────────┐
│                       nano-vLLM 知识图谱                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  用户接口层                                                       │
│  ├── llm.py           LLM 类（对外唯一入口）                       │
│  └── example.py       端到端示例                                  │
│                                                                  │
│  引擎层                                                           │
│  ├── llm_engine.py    generate() 主循环                          │
│  │   ├── step()       单步：调度 → 推理 → 后处理                   │
│  │   ├── add_request() 添加请求（tokenize → Sequence）            │
│  │   └── generate()   完整生成循环 + tqdm 监控                    │
│  │                                                               │
│  ├── scheduler.py     Continuous Batching 调度器                  │
│  │   ├── waiting      等待队列（新请求）                           │
│  │   ├── running      运行队列（正在生成）                         │
│  │   ├── schedule()   决定本轮处理谁                              │
│  │   └── postprocess() 更新状态 + 释放资源                        │
│  │                                                               │
│  └── model_runner.py  模型执行器                                  │
│      ├── prepare_prefill()  准备 Prefill 输入                      │
│      ├── prepare_decode()   准备 Decode 输入                       │
│      ├── run_model()        模型前向（含 CUDA Graph 分支）          │
│      └── run()              完整 step 流程                        │
│                                                                  │
│  模型层                                                           │
│  ├── models/qwen3.py    Qwen3ForCausalLM                         │
│  │   ├── Qwen3Model           Transformer 主干                    │
│  │   │   ├── embed_tokens     词嵌入                              │
│  │   │   ├── DecoderLayer ×28 每层包含 Attention + MLP            │
│  │   │   └── RMSNorm          最终归一化                           │
│  │   ├── Qwen3Attention       注意力层                            │
│  │   │   ├── qkv_proj   QKV 融合投影（Column Parallel）            │
│  │   │   ├── q_norm/k_norm    Q/K 归一化（Qwen3 特有）             │
│  │   │   ├── rotary_emb       RoPE 旋转位置编码                    │
│  │   │   ├── attn             PagedAttention (FlashAttention)     │
│  │   │   └── o_proj           O 投影（Row Parallel）               │
│  │   ├── Qwen3MLP             SwiGLU 前馈网络                     │
│  │   │   ├── gate_up_proj     Gate+Up 融合投影                     │
│  │   │   ├── SiluAndMul       SwiGLU 激活                         │
│  │   │   └── down_proj        Down 投影                           │
│  │   └── lm_head              vocab 投影（vocab_size 大，独立）     │
│  │                                                               │
│  └── layers/             模型组件库                                │
│      ├── layernorm.py    RMSNorm（含残差融合）                     │
│      ├── activation.py   SwiGLU                                  │
│      ├── rotary_embedding.py RoPE                                │
│      ├── linear.py       QKV/Merged/Row Linear + TP               │
│      ├── attention.py    PagedAttention + Triton kernel           │
│      └── sampler.py      采样器（Gumbel-Max）                      │
│                                                                  │
│  数据层                                                           │
│  ├── engine/sequence.py   Sequence 状态机                         │
│  │   ├── token_ids        完整 token 序列                         │
│  │   ├── block_table      逻辑块→物理页映射                        │
│  │   └── status           WAITING → RUNNING → FINISHED           │
│  ├── engine/block_manager.py  Block 管理器                        │
│  │   ├── free_block_ids   空闲物理页池                            │
│  │   ├── allocate()       分配（含 Prefix Cache 命中检查）         │
│  │   ├── append_slot()    追加 slot（Decode 阶段）                 │
│  │   └── deallocate()     释放                                    │
│  └── utils/context.py     全局 Context（跨层传元数据）              │
│                                                                  │
│  配置层                                                           │
│  ├── config.py           全局配置                                  │
│  └── sampling_params.py  采样参数（per-request）                   │
│                                                                  │
│  工具层                                                           │
│  └── utils/loader.py     权重加载（safetensors → 融合映射）        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 你已经掌握的核心知识点

1. **PagedAttention**：类比操作系统虚拟内存，把 KV Cache 切成固定大小的页，消除显存碎片
2. **Continuous Batching**：双队列调度（waiting + running），每轮动态决定处理哪些请求
3. **Prefix Cache**：通过 xxhash 做块级哈希，相同前缀的请求共享物理页
4. **KV Cache 生命周期**：Prefill 阶段写入整段 prompt 的 K/V，Decode 阶段追加单个 token
5. **全局 Context 模式**：避免在模型层之间逐层传递元数据
6. **融合权重加载**：HF 的分离权重（q_proj/k_proj/v_proj）通过 packed_modules_mapping 和 weight_loader 协议映射到本地融合层
7. **FlashAttention 双模式**：varlen（Prefill）+ with_kvcache（Decode）
8. **Gumbel-Max Trick**：用 `argmax(logits / Exp(1))` 等价于 softmax 采样
9. **Triton kernel**：手写 store_kvcache_kernel 高效写入 KV Cache
10. **Qwen3 特色**：GQA（Grouped Query Attention）、Q/K Norm、Pre-Norm 架构

### 4.3 和原版 nano-vLLM（14K stars）的对比

| 特性 | 原版 nano-vLLM | 本复刻版本 |
|------|---------------|-----------|
| PagedAttention | ✅ | ✅ |
| Continuous Batching | ✅ | ✅ |
| Prefix Cache (hash) | ✅ | ✅ |
| FlashAttention | ✅ | ✅ |
| Triton KV Cache kernel | ✅ | ✅ |
| CUDA Graph | ✅ | ✅ (Day7 教学版) |
| Tensor Parallel | ✅ | ✅ (Day7 教学版) |
| Chunked Prefill | ✅ | ❌ (进阶) |
| Radix Prefix Cache | ✅ | ❌ (进阶) |
| Speculative Decoding | ❌ | ❌ |
| MoE | ❌ | ❌ |
| FP8 量化 | ✅ | ❌ |

### 4.4 下一步可以探索的方向

1. **Radix Tree Prefix Cache**（比 hash 方案更灵活的前缀树匹配）
2. **Chunked Prefill**（长 prompt 分块处理，提升调度公平性）
3. **FP8 KV Cache 量化**（把 KV Cache 从 fp16 压缩到 fp8，节省一半显存）
4. **Streaming 输出**（逐 token 实时返回，不等到全部生成完）
5. **Beam Search**（多条候选序列同时探索）

---

## 5. 验证命令

```bash
cd nano_vll_repro

# 验证 CUDA Graph 相关代码（需要 CUDA）
python -m py_compile layers/linear.py engine/model_runner.py

# 阶段 1: 测试单卡 fallback（TP 代码在 world_size=1 时退化为单卡）
python tests/test_Day4.py
```

> **💡 TP 与 Day1-6 代码的衔接说明：**
>
> Day7 的 `layers/linear.py` 引入了 `ColumnParallelLinear` 和 `RowParallelLinear` 作为
> `QKVLinear`/`MergedLinear`/`RowLinear` 的**替代**。两者的关系是：
>
> - **单卡模式**（`tensor_parallel_size=1`）：TP 代码退化为普通 Linear，行为与 Day2 完全等价
> - **替换方式**：用 Day7 的 `linear.py` **整体替换** Day2 的版本，同时更新 `models/qwen3.py` 的 import
> - **不替换也可以**：如果只跑单卡，Day2 的 `QKVLinear`/`MergedLinear`/`RowLinear` 完全够用
>
> 同理，`models/qwen3.py` 和 `engine/model_runner.py` 的 Day7 版本增加了 TP 支持，
> 但单卡模式下行为不变。建议在 Day1-6 全部跑通后再应用 Day7 的改动。

# 阶段 2: 端到端推理（验证 graph 录制不干扰正常流程）
# 如果显存不够录制 graph，可以先加 enforce_eager=True
python -c "
from llm import LLM
llm = LLM('models/Qwen3-0.6B', enforce_eager=True)
print('单卡 eager 模式正常')
"

# 阶段 3: 如果有多张 GPU，测试 TP
# torchrun --nproc_per_node=2 tests/test_Day4.py
```

---

## 附录：与参考仓库的对照

- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（14K stars）：本项目的直接参考，~1200 行 Python
- [vllm-project/vllm](https://github.com/vllm-project/vllm)（82K stars）：生产级推理引擎，2000+ contributors
- vLLM 论文 (SOSP'23)：PagedAttention 原始设计
- FlashAttention 论文：IO-aware 注意力算法
- RoPE 论文：旋转位置编码

---

**恭喜！你已完整掌握 nano-vLLM 的核心架构。** 🎉
