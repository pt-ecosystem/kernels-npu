# Introducing kernels-npu: A component for accelerating Transformers on NPU

## 写在前面：kernels-npu是什么
Kernels-npu 允许 Transformers 库（理论上所有 Python 模型库都可以）直接从 [HuggingFace-Hub](https://huggingface.co/) 动态加载计算内核。HuggingFace-Hub 加载内核与传统的直接使用 Python 计算内核的区别在于其具备以下特性：

- 易移植：从 PYTHONPATH 之外的路径加载内核。你不必再针对每个依赖 Transformers 的上层库中做MonkeyPatch。
- 版本的扩展性：你可以为同一 Python 模块加载某一内核的多个版本。
- 版本的兼容性：kernels 为加载 HuggingFace-Hub 中的计算内核制定了一套标准文件路径命名。该命名使用torch, cuda/cann, ABIs, linux name 和 os作为关键字。这使得在向 HuggingFace-Hub 贡献时，必须保证计算内核在特定关键字排列组合下对应版本的兼容性。

请注意：[kernels-npu](https://github.com/huggingface/kernels) 指的是原生支持 npu 能力后的 kernels 工具，并没有额外的 kernels-npu 的工具。这样命名只是为了方便行文。

## 快速了解kernels使用方式

### 通过`use_kernel_forward_from_hub()`装饰器

1.现在你有一个transformers中的计算模块，比如 `class Qwen3RMSNorm(nn.Module)` [-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L50-L67)。 如果我想加速这部分，只需要满足两个条件。

- 有 `forward` 函数
- npu 有办法对 forward 中的逻辑加速

审视一下发现 `Qwen3RMSNorm` 满足这两个条件：forward 存在，且我们有融合算子 `torch_npu.npu_rms_norm()` 。

2.在 HuggingFace-Hub 的 kernels-ext-npu 组织中创建一个 RMSNorm 加速计算内核[专用的卡片](https://huggingface.co/kernels-ext-npu/RMSNorm)。

3.根据命名规则在 RMSNorm 卡片下创建（断句）你的计算内核支持的（断句）以依赖版本命名的（断句）文件夹。

命名规则有两种：通用命名规则和严格限制版本命名规则。

严格限制版本的文件夹目录树如下，创建时必须满足这个规则：`"torch{torch_version.major}{torch_version.minor}-{cxxabi}-{compute_framework}-{cpu}-{os}"` ，它在 src/kernels/utils.py 中的 `build_variant()` 被定义。 [-附源码](https://github.com/huggingface/kernels/blob/main/src/kernels/utils.py#L46-L80)

```
.
└── RMSNorm
    ├── README.md
    └── build
        ├── torch26-cxx11-cann82-aarch64-linux
        │   └── RMSNorm
        │       ├──  __init__.py
        │       └──  layers.py
        └── torch27-cxx11-cann82-aarch64-linux
            └── RMSNorm
                ├──  __init__.py
                └──  layers.py
```

通用命名规则自然就是指不严格限制版本，它的目录树如下。 在src/kernels/utils.py中的 `universal_build_variant()` 被定义。[-附源码](https://github.com/huggingface/kernels/blob/main/src/kernels/utils.py#L83-L85)

```
.
└── RMSNorm
    ├── README.md
    └── build
        └── torch-universal
            └── RMSNorm
                ├──  __init__.py
                └──  layers.py
```

4.在 layers.py 文件中用 `torch_npu.npu_rms_norm()` 写一个 `forward`。如：

```
import torch
import torch_npu


class RMSNorm(torch.nn.Module):
    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]
```

至此 kernels 侧所需要的工作都已准备完成。接下来是 transformers 侧的工作价绍。

5.在 transformers 的 src/transformers/integrations/hub_kernels.py 中注册该计算内核。[-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py#L49)

```
"RMSNorm": {
    "npu": LayerRepository(
        repo_id="kernels-ext-npu/RMSNorm",
        layer_name="RMSNorm",
    )
```

6.在 transformers 中被替换的模块上添加装饰器。[-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L49)

```
@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass
```

7.kernelize化模型

8.找到 AutoModel ，将 use_kernels 参数设置为 True ，即可完成计算逻辑的替换；将 logging.basicConfig() 设置为l evel=logging.DEBUG 即可通过打屏日志看到RMSNorm替换内核是否生效。

### 使用xx方式



## FAQ 

### 为什么只替换forward部分

https://github.com/huggingface/kernels/pull/153
