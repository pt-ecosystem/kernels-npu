# install_nix_and_start_build_on_npu

## 配置环境
### 下载并安装nix
参考https://mirrors.tuna.tsinghua.edu.cn/help/nix/

### 重启shell
重启shell后nix才会生效

### 声明路径
`export PATH=$PATH:/nix/var/nix/profiles/default/bin/`

### 验证nix是否生效
`nix --version`

## （可选的）下载并安装cachix
期间会下载一些安装包，这里要注意下网络问题
参考https://docs.cachix.org/installation

## kernels-builder中
### 下载kernels
`git clone https://github.com/huggingface/kernel-builder.git`

### 使能Hugging Face binary cache
`nix run nixpkgs#cachix --extra-experimental-features nix-command --extra-experimental-features flakes --use huggingface`

### build
`nix build . --extra-experimental-features flakes --extra-experimental-features nix-command --override-input kernel-builder github:huggingface/kernel-builder --max-jobs 8 -j 8 -L`
