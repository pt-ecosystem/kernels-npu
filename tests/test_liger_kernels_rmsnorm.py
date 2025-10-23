# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import torch
import torch.nn as nn
from typing import Union
import logging
import time

from kernels import (
    Device,
    LayerRepository,
    Mode,
    register_kernel_mapping,
    use_kernel_forward_from_hub,
    kernelize,
)

_kernels_available = True

# Setting the level to DEBUG will show which kernels are being used.
# logging.basicConfig(level=logging.DEBUG)

DEVICE = "cuda" if torch.cuda.is_available() else "npu"

_KERNEL_MAPPING: dict[str, dict[Union[Device, str], LayerRepository]] = {
    "RMSNorm": {
        "cuda": LayerRepository(
            repo_id="kernels-community/liger_kernels",
            layer_name="LigerRMSNorm",
        )
    }
}

register_kernel_mapping(_KERNEL_MAPPING)


# PyTorch reference implementation of RMSNorm.
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(DEVICE)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# # Compiled version for performance comparison
compiled_torch_rmsnorm = torch.compile(Qwen3RMSNorm)


# triton RMSNorm kernel
@use_kernel_forward_from_hub("RMSNorm")
class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(DEVICE)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return NotImplementedError("This method will be replaced by the kernel from hub.")


def test_rmsnorm(s1, s2, s3, hidden_size=1024, eps=1e-5):
    x = torch.randn(s1, s2, s3, hidden_size, dtype=torch.float32, device=DEVICE)
    weight = torch.rand(hidden_size, dtype=torch.float32, device=DEVICE)

    # PyTorch reference implementation
    torch_rmsnorm = Qwen3RMSNorm(hidden_size, eps).to(DEVICE)
    start_time = time.time()
    torch_res = torch_rmsnorm(x)
    print(f"torch_rmsnorm time: {time.time() - start_time}")

    # Compiled version
    compiled_rmsnorm = compiled_torch_rmsnorm(hidden_size, eps).to(DEVICE)
    start_time = time.time()
    compiled_res = compiled_rmsnorm(x)
    print(f"compiled_rmsnorm time: {time.time() - start_time}")
    
    # Triton RMSNorm kernel
    triton_rmsnorm = TritonRMSNorm(hidden_size, eps).to(DEVICE)
    kernelize(triton_rmsnorm, device=DEVICE, mode=Mode.INFERENCE)
    start_time = time.time()
    triton_res = triton_rmsnorm(x)
    print(f"triton_rmsnorm time: {time.time() - start_time}")

    assert torch.allclose(compiled_res, torch_res, atol=1e-2, rtol=0.0)
    assert torch.allclose(triton_res, torch_res, atol=1e-2, rtol=0.0)
    print(f"-----------------------shape [{s1}, {s2}, {s3}, {hidden_size}] RMSNorm test passed!-----------------------")


if __name__ == "__main__":
    test_rmsnorm(1, 1, 1, 1)
    test_rmsnorm(8, 16, 32, 1024)
    test_rmsnorm(4, 8, 16, 4096)
    test_rmsnorm(2, 4, 8, 8192)
    test_rmsnorm(1, 2, 4, 16384)
    test_rmsnorm(1, 1, 2, 32768)
    test_rmsnorm(1, 16, 32, 65536)
