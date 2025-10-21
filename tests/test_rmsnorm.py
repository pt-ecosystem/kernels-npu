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

# Reference from: https://github.com/kapilsh/gpt-oss-scratch/blob/main/kernels/rms_norm.py


"""
RMS Normalization Triton Kernel
===============================

Optimized Triton implementation of RMS Normalization with forward and backward passes.

RMS Normalization computes:
y = (x / sqrt(mean(x^2) + eps)) * weight
"""

import torch
import kernels
import triton
import torch.nn.functional as F
import triton.language as tl
from loguru import logger
import time


# Configure torch settings for compiled functions
try:
    torch._functorch.config.donated_buffer = False  # type: ignore
except AttributeError:
    # Handle cases where _functorch is not available
    pass


# Setup
torch.manual_seed(42)
if torch.version.cuda is not None and torch.cuda.is_available():
    DEVICE = "cuda"
    triton_rmsnorm = kernels.get_kernel("kernels-ext-npu/triton_rmsnorm").RMSNorm
elif hasattr(torch._C, "_get_privateuse1_backend_name"):
    DEVICE = torch._C._get_privateuse1_backend_name()
    triton_rmsnorm = kernels.get_kernel("kernels-ext-npu/triton_rmsnorm").RMSNorm
    torchnpu_rmsnorm = kernels.get_kernel("kernels-ext-npu/rmsnorm")
else:
    raise RuntimeError("No supported device found for testing rmsnorm.")


# PyTorch reference implementation of RMSNorm.
def pytorch_rmsnorm(x, weight, eps=1e-5):
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


# Compiled version for performance comparison
pytorch_rmsnorm_compiled = torch.compile(pytorch_rmsnorm)


def test_rmsnorm(
    batch_size=1151, feature_dim=8192, dtype=torch.float16, eps=1e-5, device=None
):
    if device is None:
        device = DEVICE

    # Create test data
    weight = torch.rand(feature_dim, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(batch_size, feature_dim, dtype=dtype, device=device)

    # Forward pass comparison
    logger.info("Testing RMSNorm performance...")
    start_time = time.time()
    y_triton = triton_rmsnorm(x, (feature_dim,), weight, eps)  # type: ignore
    logger.info(f"triton_time: {time.time() - start_time}")

    start_time = time.time()
    y_pytorch = pytorch_rmsnorm(x, weight, eps)
    logger.info(f"native_torch_time: {time.time() - start_time}")

    start_time = time.time()
    y_compiled = pytorch_rmsnorm_compiled(x, weight, eps)
    logger.info(f"compiled_torch_time: {time.time() - start_time}")

    if DEVICE == "npu":
        start_time = time.time()
        y_torchnpu = torchnpu_rmsnorm(x, weight, eps)
        logger.info(f"torch_npu__time: {time.time() - start_time}")

    logger.success("All performance tests passed!")

    # Assertions
    logger.info("Testing RMSNorm correctness...")
    assert torch.allclose(
        y_triton, y_pytorch, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs Native PyTorch"
    assert torch.allclose(
        y_triton, y_compiled, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs Compiled PyTorch"

    if DEVICE == "npu":
        assert torch.allclose(
            y_triton, y_torchnpu, atol=1e-2, rtol=0
        ), "Forward pass mismatch: Triton vs torch_npu"

    logger.success("All correctness tests passed!")


if __name__ == "__main__":
    test_rmsnorm()
