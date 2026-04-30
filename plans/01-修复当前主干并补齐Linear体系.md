# 01. 修复 Linear 和 Loader 的加载协议

这一篇先不做 Tensor Parallel。

这一篇只做一件事：

> 把当前单卡版 `QKVLinear`、`MergedLinear`、`RowLinear` 和 `utils/loader.py` 的权重加载协议讲清楚、补完整、测住。

改完后要达到这几个结果：

1. `QKVLinear / MergedLinear / RowLinear` 的参数都能通过 `weight_loader` 加载权重。
2. `QKVLinear` 如果启用 bias，bias 也有 `weight_loader`。
3. 权重复制前会统一对齐 `device` 和 `dtype`。
4. `loader.py` 只负责“分发权重”，不在里面硬写 QKV 的内部布局。
5. `tests/test_Day4.py` 能覆盖 QKV、Gate-Up、Row 三类 Linear。

Tensor Parallel 放到 `05`，不要在这里提前做。

---

## 1. 当前代码是什么状态

当前 `layers/linear.py` 已经有三个类：

- `QKVLinear`
- `MergedLinear`
- `RowLinear`

所以本篇不是“补类名”，而是修协议细节。

当前主要问题有四个：

1. `QKVLinear.weight` 有 `weight_loader`，但 `QKVLinear.bias` 没有。
2. 几个 loader 都直接 `copy_`，没有先把加载进来的权重转到目标参数的 `device` 和 `dtype`。
3. `loader.py` 里 packed 权重和普通权重的分发逻辑能用，但说明不够清楚。
4. 测试没有完整覆盖 `RowLinear`。

---

## 2. 先记住这条协议

权重加载只有两种情况。

第一种：普通参数。

```python
weight_loader(param, loaded_weight)
```

第二种：融合参数，比如 HF 里是 `q_proj / k_proj / v_proj`，本仓库里是一个 `qkv_proj`。

```python
weight_loader(param, loaded_weight, shard_id)
```

`shard_id` 用来告诉融合层：这块权重应该写到 Q、K、V 的哪个位置。

`loader.py` 不应该理解 QKV 的内部布局。它只负责找到参数，然后调用参数自己的 `weight_loader`。

---

## 3. 修改 `layers/linear.py`

### 3.1 增加一个安全复制函数

放在 `layers/linear.py` 里，建议放到几个类定义前或 `default_weight_loader` 前。

```python
def copy_weight_to_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    把读取到的权重写入目标参数。

    为什么不直接 param.data.copy_(loaded_weight)：
    1. safetensors 读出来的权重通常在 CPU 上。
    2. 模型参数可能已经在 CUDA 上。
    3. 模型参数可能是 fp16 / bf16，而加载权重可能是 fp32。

    所以复制前先对齐 device 和 dtype。
    """
    loaded_weight = loaded_weight.to(device=param.device, dtype=param.dtype)
    param.data.copy_(loaded_weight)
```

如果是写入参数的一段切片，也按同样规则先转换：

```python
target = loaded_weight.to(device=param.device, dtype=param.dtype)
param.data[start:end].copy_(target)
```

### 3.2 给 `QKVLinear.bias` 也绑定 loader

在 `QKVLinear.__init__()` 里，原来只有：

```python
self.weight.weight_loader = self._weight_loader
```

改成：

```python
self.weight.weight_loader = self._weight_loader

# 如果启用了 bias，它和 weight 一样也是融合布局。
# 比如 bias 也会按 [q_bias, k_bias, v_bias] 拼在一起。
# 所以它必须使用同一个 _weight_loader。
if self.bias is not None:
    self.bias.weight_loader = self._weight_loader
```

### 3.3 修改 QKV 的加载函数

核心逻辑是：先根据 `shard_id` 找到写入区间，再复制。

```python
def _weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
) -> None:
    """
    把 HF 的 q_proj / k_proj / v_proj 权重写进本地 qkv_proj。

    param:
        目标参数，可能是 qkv_proj.weight，也可能是 qkv_proj.bias。

    loaded_weight:
        HF 中读取出来的一块权重，比如 q_proj.weight。

    shard_id:
        "q" 表示写入 Q 区间；
        "k" 表示写入 K 区间；
        "v" 表示写入 V 区间。
    """
    if shard_id == "q":
        start = 0
        size = self.q_size
    elif shard_id == "k":
        start = self.q_size
        size = self.kv_size
    elif shard_id == "v":
        start = self.q_size + self.kv_size
        size = self.kv_size
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")

    end = start + size

    # loaded_weight 先转到 param 所在设备和 dtype，再写入目标区间。
    target = loaded_weight.to(device=param.device, dtype=param.dtype)
    param.data[start:end].copy_(target)
```

### 3.4 修改 `MergedLinear` 和 `RowLinear`

`MergedLinear` 的写法也要先对齐 dtype/device：

```python
def _weight_loader(
    self,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int,
) -> None:
    """
    把 gate_proj 或 up_proj 写入 gate_up_proj。

    shard_id=0 表示 gate_proj；
    shard_id=1 表示 up_proj。
    """
    start = shard_id * self.output_size
    end = start + self.output_size
    target = loaded_weight.to(device=param.device, dtype=param.dtype)
    param.data[start:end].copy_(target)
```

`RowLinear` 当前还是单卡版本，所以直接复制即可：

```python
def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    RowLinear 在 01 阶段还是普通单卡 Linear。

    这里不切分输入维，也不 all_reduce。
    真正的 row parallel 到 05 再做。
    """
    copy_weight_to_param(param, loaded_weight)
```

`default_weight_loader` 也改成统一 helper：

```python
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """
    普通参数使用的默认加载器。

    典型例子：
    - embedding.weight
    - norm.weight
    - lm_head.weight
    """
    assert param.data.shape == loaded_weight.shape, (
        f"Shape mismatch: {param.data.shape} vs {loaded_weight.shape}"
    )
    copy_weight_to_param(param, loaded_weight)
```

---

## 4. 修改 `utils/loader.py`

这里不要把 loader 写成“懂所有层内部结构”的大函数。

它只做三件事：

1. 遍历 safetensors 文件。
2. 判断某个 HF 权重是否属于 packed 参数。
3. 找到目标参数，调用目标参数自己的 `weight_loader`。

推荐把 safetensors 文件先排序，方便复现调试日志：

```python
safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
```

核心循环可以整理成下面这样：

```python
for file_path in safetensor_files:
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for weight_name in f.keys():
            # loaded_weight 是 HF 原始权重。
            # 这里先放 CPU，真正复制时再由各参数的 weight_loader 搬到目标设备。
            loaded_weight = f.get_tensor(weight_name)

            for original_name, (packed_name, shard_id) in packed_modules_mapping.items():
                if original_name not in weight_name:
                    continue

                # 例子：
                # HF: model.layers.0.self_attn.q_proj.weight
                # 本地: model.layers.0.self_attn.qkv_proj.weight
                param_name = weight_name.replace(original_name, packed_name)
                param = model.get_parameter(param_name)

                # packed 参数比普通参数多一个 shard_id。
                # loader.py 不关心 q/k/v 的切片位置，交给参数自己的 loader。
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_count += 1
                break
            else:
                # 没命中 packed mapping，就按普通参数加载。
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_count += 1
```

这段代码里最重要的是 `for ... else`：

- `break` 表示已经按 packed 参数加载。
- 没有 `break` 才走普通参数加载。

---

## 5. 修改 `tests/test_Day4.py`

Day4 测试至少补三类断言。

### 5.1 参数必须挂有 `weight_loader`

```python
qkv = QKVLinear(512, num_heads=8, num_kv_heads=2, head_dim=64, bias=True)
merged = MergedLinear(512, 1024, num_shards=2)
row = RowLinear(512, 256)

assert hasattr(qkv.weight, "weight_loader")
assert hasattr(qkv.bias, "weight_loader")
assert hasattr(merged.weight, "weight_loader")
assert hasattr(row.weight, "weight_loader")
```

### 5.2 `RowLinear` 要测加载和前向

```python
row = RowLinear(512, 256, bias=False)

# 用随机权重模拟从 HF 文件读出来的权重。
loaded_weight = torch.randn(256, 512)
row.weight.weight_loader(row.weight, loaded_weight)

x = torch.randn(4, 512)
y = row(x)

assert torch.allclose(row.weight.data, loaded_weight)
assert y.shape == (4, 256)
```

### 5.3 `MergedLinear` 使用当前真实签名

当前真实签名是：

```python
MergedLinear(input_size=512, output_size=1024, num_shards=2)
```

不要写成：

```python
MergedLinear(512, [1024, 1024])
```

后者不是当前仓库的接口。

---

## 6. 验收命令

先做语法检查：

```bash
python -m py_compile layers/linear.py utils/loader.py tests/test_Day4.py
```

再跑 Day4：

```bash
python tests/test_Day4.py
```

如果导入阶段因为 `flash_attn` 缺失失败，先安装依赖或换到有依赖的环境再测。

---

## 7. 常见坑

1. **在 01 里提前引入 TP 类名**
   不要在这一篇加入 `QKVParallelLinear`、`RowParallelLinear`。那是 05 的内容。

2. **只给 weight 绑定 loader，忘了 bias**
   默认配置可能不开 bias，所以这个 bug 容易潜伏很久。

3. **裸 `copy_` 不转 dtype/device**
   CPU fp32 权重直接复制到 CUDA bf16 参数，很容易出问题。

4. **让 `loader.py` 理解 QKV 内部布局**
   这会让 loader 越写越乱。布局应该由 Linear 层自己负责。

---

## 8. 本篇结束后你应该明白

这一篇真正要学会的是：

1. HF 分离权重如何写入本地融合参数。
2. `weight_loader` 为什么要挂在参数对象上。
3. 单卡 Linear 和未来 TP Linear 的边界在哪里。

下一篇进入 Qwen3 模型主干：

- `02-补齐Qwen3模型主干与权重映射.md`
