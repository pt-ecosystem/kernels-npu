# kernels-ext-npu

Kernel sources for https://huggingface.co/kernels-ext-npu


## 介绍 kernels-npu

参考[文档](doc/Introducing_kernels_npu.md)


## 从源码中构建算子

参考[文档](doc/Install_nix_and_build_on_npu.md)


## 当前支持

| kernel          | build from                                                            | category      |
|-----------------|-----------------------------------------------------------------------|---------------|
| flash-attn2-npu | [source code](flash-attn2-npu/README.md)                              | torch_npu     |
| rmsnorm         | [source code](rmsnorm/README.md)                                      | torch_npu     |
| rmsnorm         | [source code](https://huggingface.co/kernels-community/liger_kernels) | triton-ascend |


## 其他说明
1. 目前，NPU 的主要优化手段仍是 torch_npu， triton-ascend 的优化 kernel 仅供参考。
2. triton-ascend 的安装可参考[文档](doc/Install_triton_ascend.md)。
