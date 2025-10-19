"""
RMS Normalization Triton Kernel
===============================

Optimized Triton implementation of RMS Normalization with forward and backward passes.

RMS Normalization computes:
y = (x / sqrt(mean(x^2) + eps)) * weight
"""

# Copied from: https://github.com/kapilsh/gpt-oss-scratch/blob/main/kernels/rms_norm.py

import torch
import triton
import torch.nn.functional as F
import triton.language as tl
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

# Configure torch settings for compiled functions
try:
    torch._functorch.config.donated_buffer = False  # type: ignore
except AttributeError:
    # Handle cases where _functorch is not available
    pass

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _rms_norm_fwd_fused(
    input_ptr,  # pointer to the input tensor
    output_ptr,  # pointer to the output tensor
    weight_ptr,  # pointer to the weight tensor
    rstd_ptr,  # pointer to the reciprocal standard deviation tensor
    row_stride,  # stride for moving to the next row
    feature_dim,  # number of features (columns) in input
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of input and output tensors to compute
    row_idx = tl.program_id(0)
    output_ptr += row_idx * row_stride
    input_ptr += row_idx * row_stride

    # Compute variance (mean of squared values for RMS)
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_values = tl.load(
            input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0
        ).to(tl.float32)
        sum_of_squares += input_values * input_values

    variance = tl.sum(sum_of_squares, axis=0) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)

    # Store reciprocal standard deviation for backward pass
    tl.store(rstd_ptr + row_idx, reciprocal_std)

    # Normalize input and apply weight transformation
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_indices < feature_dim

        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask)
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0.0).to(
            tl.float32
        )

        normalized_values = input_values * reciprocal_std
        output_values = normalized_values * weight_values

        # Write final output
        tl.store(output_ptr + col_indices, output_values, mask=valid_mask)


class RMSNorm(torch.autograd.Function):
    """
    Triton-optimized RMS Normalization with automatic differentiation support.
    """

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        # Allocate output tensor
        y = torch.empty_like(x)

        # Reshape input to 2D for processing
        x_reshaped = x.reshape(-1, x.shape[-1])
        batch_size, feature_dim = x_reshaped.shape
        rstd = torch.empty((batch_size,), dtype=torch.float32, device=x.device)

        # Determine optimal block size (limited by 64KB per feature)
        max_fused_size = 65536 // x.element_size()
        BLOCK_SIZE = min(max_fused_size, triton.next_power_of_2(feature_dim))

        if feature_dim > BLOCK_SIZE:
            raise RuntimeError("This RMS norm doesn't support feature dim >= 64KB.")

        # Heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # Launch forward kernel
        _rms_norm_fwd_fused[(batch_size,)](  # type: ignore
            x_reshaped,
            y,
            weight,
            rstd,
            x_reshaped.stride(0),
            feature_dim,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
        )
        return y


# Convenience function for using the optimized RMS norm
rms_norm = RMSNorm.apply


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
