## Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

DEVICE = "npu"

_KERNEL_MAPPING: dict[str, dict[Union[Device, str], LayerRepository]] = {
    "RMSNorm": {
        DEVICE: LayerRepository(
            repo_id="kernels-ext-npu/RMSNorm",
            layer_name="RMSNorm",
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


# torch_npu RMSNorm kernel
@use_kernel_forward_from_hub("RMSNorm")
class TorchNPURMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(DEVICE)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return NotImplementedError("This method will be replaced by the kernel from hub.")


def test_rmsnorm(s1, s2, s3, hidden_size=1024, eps=1e-5):
    torch.manual_seed(42)
    x = torch.randn(s1, s2, s3, hidden_size, dtype=torch.float32, device=DEVICE)
    weight = torch.rand(hidden_size, dtype=torch.float32, device=DEVICE)

    # PyTorch reference implementation
    torch_rmsnorm = Qwen3RMSNorm(hidden_size, eps).to(DEVICE)
    start_time = time.time()
    torch_res = torch_rmsnorm(x)
    print(f"torch_rmsnorm time: {time.time() - start_time}")

    # torch_npu RMSNorm kernel
    torch_npu_rmsnorm = TorchNPURMSNorm(hidden_size, eps).to(DEVICE)
    kernelize(torch_npu_rmsnorm, device=DEVICE, mode=Mode.INFERENCE)
    start_time = time.time()
    torch_npu_res = torch_npu_rmsnorm(x)
    print(f"torch_npu_rmsnorm time: {time.time() - start_time}")

    assert torch.allclose(torch_npu_res, torch_res, atol=1e-2, rtol=0.0)
    print(f"-----------------------shape [{s1}, {s2}, {s3}, {hidden_size}] RMSNorm test passed!-----------------------")


if __name__ == "__main__":
    test_rmsnorm(1, 1, 1, 1)
    print("The data from the first test case can be disregarded as it is unreliable.")
    
    test_rmsnorm(1, 1, 1024)
    test_rmsnorm(1, 1, 1024)
    test_rmsnorm(1, 1, 8, 128)
    test_rmsnorm(1, 1, 8, 128)
    test_rmsnorm(1, 1, 16, 128)
    test_rmsnorm(1, 1, 16, 128)
