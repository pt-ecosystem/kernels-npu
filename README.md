# kernels-transformers-ascend

## 说明

为了方便项目管理，这里将作为[kernels-ext-npu](https://huggingface.co/kernels-ext-npu)组织代码更新的入口。

## 项目目录

```shell
.
├── LICENSE
├── README.md
├── install_triton_ascend.md
└── kernels-ext-npu
    ├── RMSNorm
    │   ├── README.md
    │   └── build
    │       └── torch-universal
    │           └── RMSNorm
    │               ├── __init__.py
    │               └── layers.py
    └── SwiGlu
        ├── README.md
        └── build
            ├── torch26-cxx11-cann82-aarch64-linux
            │   └── SwiGlu
            │       ├── __init__.py
            │       └── layers.py
            └── torch27-cxx11-cann82-aarch64-linux
                └── SwiGlu
                    ├── __init__.py
                    └── layers.py
```
## 如何修改/贡献kernels到kernels-ext-npu

1.下载时添加--recurse-submodules参数，即可同时下载kernels-ext-npu文件夹下关联项目。

`git clone --recurse-submodules https://github.com/pt-ecosystem/kernels-transformers-ascend`


2.修改时在kernels-ext-npu文件夹下对应项目中修改并push到hub。


3.贡献新的kernels时，需要在[kernels-ext-npu](https://huggingface.co/kernels-ext-npu)组织中按对应规则新建，并使用submodule参数关联。

`git submodule add <repository-url> <path-in-parent-repo>`