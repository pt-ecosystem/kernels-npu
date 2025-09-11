# trton-ascend安装指南

## 使用平台
https://www.autodl.com/

## 主要参考的文档
https://gitee.com/ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md

## 补充操作
1. CANN严格使用文档指定版本，即CANN>=8.2.RC1.alpha003版本，否则会出现“找不到npuc目录”的报错。
2. 可通过`bash Ascend-cann-toolkit --install --install-path=/home/install-cann`自定义安装CANN的路径，但要确保有执行权限。这里推荐/home目录。
3. 在`Ascend-cann-kernels --install --install-path=/home/install-cann`时同理。
4. source /home/install-cann/xx/set_env.sh，切记。
5. apt安装clang。pip安装pythonu依赖、torch、torch_npu，无事发生。
6. llvm严格使用commit ID: `b5cc222d7429fe6f18c787f633d5262fac2e676f`，测试main分支报错。
7. 等待漫长的clang构建安装LLVM过程。GCC编译没试过是否能成功。
8. 克隆triton-ascend时注意同时克隆的ascendnpu-ir和triton的commit ID，测试各自的main分支报错。
9. 等待漫长的triton-ascend构建安装过程。
10. 报错：“/lib64/libstdc++.so.6中缺少glibcxx_3.4.30版本”。参考[博客](https://blog.csdn.net/L1481333167/article/details/137919464)建立链接。
