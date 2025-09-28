# Introducing kernels-npu: A component for accelerating Transformers on NPU

## 写在前面：kernels-npu是什么

[Kernels-npu](https://github.com/huggingface/transformers) 允许 [transformers](https://github.com/huggingface/transformers) 库（理论上所有 Python 模型库都可以）直接从 [HuggingFace-Hub](https://huggingface.co/) 动态加载 npu 计算内核。HuggingFace-Hub 加载内核与传统的直接使用 Python 计算内核的区别在于其具备以下特性：

- 易移植：从 PYTHONPATH 之外的路径加载内核。你不必再针对每个依赖 Transformers 的上层库中做MonkeyPatch。
- 版本的扩展性：你可以为同一 Python 模块加载某一内核的多个版本。
- 版本的兼容性：kernels 为加载 HuggingFace-Hub 中的计算内核制定了一套标准文件路径命名。该命名使用torch, cuda/cann, ABIs, linux name 和 os作为关键字。这使得在向 HuggingFace-Hub 贡献时，必须保证计算内核在特定关键字排列组合下对应版本的兼容性。

transformers 在 v4.54.0 的 release 中首次介绍了 kernels 的集成，并将后续计算加速内核的支持都放在了这里。如 GPT-OSS 的flash-attention-3 就是通过 kernels 支持的。 

请注意：[kernels-npu](https://github.com/huggingface/kernels) 指的是原生支持 npu 能力后的 kernels 工具，并没有额外的 kernels-npu 的工具。这样命名只是为了方便行文。

## 快速了解kernels使用方式

### 通过`use_kernel_forward_from_hub()`装饰器，以RMSNorm加速为例

1.现在你有一个transformers中的计算模块，比如 `class Qwen3RMSNorm(nn.Module)` [-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L50-L67)。 如果我想加速这部分，只需要满足两个条件。

- 有 `forward` 函数
- npu 有办法对 forward 中的逻辑加速

审视一下发现 `Qwen3RMSNorm` 满足这两个条件：forward 存在，且我们有融合算子 `torch_npu.npu_rms_norm()` 。


2.在 HuggingFace-Hub 的 kernels-ext-npu 组织中创建一个 RMSNorm 加速计算内核[专用的卡片](https://huggingface.co/kernels-ext-npu/RMSNorm)。


3.根据命名规则在 RMSNorm 卡片下创建（断句）你的计算内核支持的（断句）以依赖版本命名的（断句）文件夹。

命名规则有两种：通用命名规则和严格限制版本命名规则，有一定的灵活性。

严格限制版本的文件夹目录树如下，创建时必须满足这个规则：`"torch{torch_version.major}{torch_version.minor}-{cxxabi}-{compute_framework}-{cpu}-{os}"` ，它在 kernels/src/kernels/utils.py 中的 `build_variant()` 被定义。 [-附源码](https://github.com/huggingface/kernels/blob/main/src/kernels/utils.py#L46-L80)

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

通用命名规则自然就是指不严格限制版本，它的目录树如下。 在 kernels/src/kernels/utils.py 中的 `universal_build_variant()` 被定义。[-附源码](https://github.com/huggingface/kernels/blob/main/src/kernels/utils.py#L83-L85)

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

为什么只需要写def forward，见 [FAQ](#为什么只替换forward部分)

至此 kernels 侧所需要的工作都已准备完成。接下来是 transformers 侧的工作价绍。


5.在 transformers 的 src/transformers/integrations/hub_kernels.py 中的 `_KERNEL_MAPPING` 指定你想替换的计算内核。[-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py#L49)

```
_KERNEL_MAPPING ={
    "RMSNorm": {
        "cuda": LayerRepository(
            repo_id="kernels-community/liger_kernels",
            layer_name="LigerRMSNorm",
            # revision="pure-layer-test",
        ),

        "rocm": {
            Mode.INFERENCE: LayerRepository(
                repo_id="kernels-community/liger_kernels",
                layer_name="LigerRMSNorm",
                # revision="pure-layer-test",
            )
        },

        "npu": LayerRepository(
            repo_id="kernels-ext-npu/RMSNorm",
            layer_name="RMSNorm",
        ),
    },
}


register_kernel_mapping(_KERNEL_MAPPING)
```

通过 `register_kernel_mapping` 注册该计算内核。


6.在 transformers 中被替换的模块上添加装饰器。[-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py#L49)

```
@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass
```

在`use_kernel_forward_from_hub`中的入参即是上一步操作中注册表 `_KERNEL_MAPPING` 的 key 值。此时还没有完成 forward 的替换，这步由 kernelize 函数完成。


7.kernelize 化模型

你能在 transformers/src/transformers/modeling_utils.py 中找到它。 [-附源码](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L5645-L5649)。直观上，kernelize 有主要的两个作用。

- 这是 kernels 对外暴露的、最终完成替换 forward 的部分。
- kernelize 的入参指定了 device 的类型，使模型通过 device 的指定在 `_KERNEL_MAPPING` 取到正确的替换计算内核。

为什么需要特意写一个 kernelize 函数，见 [FAQ](#为什么需要kernelize函数)


8.transformers（或依赖 transformers 的上层套件）如何使用 kernels

使用方式非常简单：
- 找到 `AutoModelxxx()` ，将 `use_kernels` 参数设置为 True ，即可完成计算逻辑的替换。
- 将 `logging.basicConfig()` 设置为 `level=logging.DEBUG` 即可通过打屏日志看到RMSNorm替换内核是否生效。


### 通过`get_kernel`方式，以FA使用为例

如果不能通过直接替换 forward 的方式使用，kernels 对外暴露了 `get_kernel` 接口[-附源码](https://github.com/huggingface/kernels/blob/main/src/kernels/utils.py#L215)。它使你仍能利用上述 step1-step4 中的操作。区别只在 step5-step8中，你需要手动指定替换。

这里贴出 transformers 的调用链，方便理解：

1.定义 model
```
from transformers import AutoModelxxx


model = AutoModelxxx()
model.set_attn_implementation("kernels-community/flash-attn3")
```

2.走到 src/transformers/modeling_utils.py

```
from .integrations.hub_kernels import is_kernel, load_and_register_kernel


def _check_and_adjust_attn_implementation():
    ···
    applicable_attn_implementation = attn_implementation
    ···
    load_and_register_kernel(applicable_attn_implementation)
    ···
```

3.走到src/transformers/integrations/hub_kernels.py
```
from kernels import get_kernel


def load_and_register_kernel():
    ···
    try:
        kernel = get_kernel(repo_id, revision=rev)
    except Exception as e:
        raise ValueError(f"An error occurred while trying to load from '{repo_id}': {e}.")
    ···
```


## FAQ 

### 为什么只替换forward部分

Q: https://github.com/huggingface/kernels/blob/main/docs/source/faq.md#why-does-kernelization-only-replace-forward-methods


### 为什么需要kernelize函数

Q: https://github.com/huggingface/kernels/blob/main/docs/source/faq.md#why-is-the-kernelization-step-needed-as-a-separate-step
