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


# Configure torch settings for compiled functions
try:
    torch._functorch.config.donated_buffer = False  # type: ignore
except AttributeError:
    # Handle cases where _functorch is not available
    pass

DEVICE = triton.runtime.driver.active.get_active_torch_device()


rms_norm_from_kernels = kernels.get_kernel("kernels-ext-npu/Triton_RMSNorm")
rms_norm = rms_norm_from_kernels.RMSNorm.apply


def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch reference implementation of RMS Norm."""
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


# Compiled version for performance comparison
pytorch_rms_norm_compiled = torch.compile(pytorch_rms_norm)


def test_rms_norm_correctness(
    batch_size=1151, feature_dim=8192, dtype=torch.float16, eps=1e-5, device=None
):
    """Test correctness of Triton RMS Norm vs PyTorch implementation."""
    if device is None:
        device = DEVICE

    # Create test data
    weight = torch.rand(feature_dim, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(batch_size, feature_dim, dtype=dtype, device=device)

    # Forward pass comparison
    y_triton: torch.Tensor = rms_norm(x, (feature_dim,), weight, eps)  # type: ignore
    y_pytorch = pytorch_rms_norm(x, weight, eps)
    y_compiled = pytorch_rms_norm_compiled(x, weight, eps)


    # Assertions
    assert torch.allclose(
        y_triton, y_pytorch, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs PyTorch"
    assert torch.allclose(
        y_triton, y_compiled, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs Compiled"

    logger.success("All correctness tests passed!")


if __name__ == "__main__":
    # Run correctness test
    logger.info("Testing RMS Norm correctness...")
    test_rms_norm_correctness()
