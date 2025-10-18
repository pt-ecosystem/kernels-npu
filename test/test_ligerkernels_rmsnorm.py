import torch
import torch.nn as nn
import torch_npu
from tritonrms.rms_norm import LigerRMSNormFunction
import sys
import time
import matplotlib.pyplot as plt


batch, hidden_size = 128, 128
seq_length = [2, 8, 16, 32, 64, 128, 256, 384]


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to("npu")
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RMSNormWithTorchNPU(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to("npu")
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


class LigerRMSNorm(Qwen3RMSNorm):
    """
    RMSNorm module that uses the optimized LigerRMSNormFunction.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.
        offset (float, optional): Offset value to shift the weight tensor. Defaults to 0.0.
        casting_mode (str, optional): The casting mode to use. Defaults to "llama".
        in_place (bool, optional): Whether to modify dY in-place to store dX during backward. Defaults to True.
    """
    

    weight: torch.Tensor
    variance_epsilon: float
    
    def forward(self, hidden_states):
        """
        Apply RMS normalization to the input tensor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, T, H) or (BxT, H)
            
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        return LigerRMSNormFunction.apply(
            hidden_states, 
            self.weight, 
            self.variance_epsilon,
            0,
            "llama",
            True
        )

def run_compare():
    torch.manual_seed(42)
    scaleout = 1
    
    for i in range(len(seq_length)):
        X = torch.randn(batch, seq_length[i], hidden_size).to("npu")
        
        native_rmsnorm = Qwen3RMSNorm(hidden_size)
        start_time_1 = time.time()
        native_output = native_rmsnorm(X)
        end_time_1 = time.time()
        print(f"native_time of seq_length={seq_length[i]}------", end_time_1 - start_time_1)


        torchnpu_rmsnorm = Qwen3RMSNormWithTorchNPU(hidden_size)
        start_time_2 = time.time()
        native_output = torchnpu_rmsnorm(X)
        end_time_2 = time.time()
        print(f"torch_npu_time of seq_length={seq_length[i]}---", end_time_2 - start_time_2)
        

        triton_rmsnorm = LigerRMSNorm(hidden_size)
        start_time_3 = time.time()
        triton_output = triton_rmsnorm(X)
        end_time_3 = time.time()
        print(f"triton_time of seq_length={seq_length[i]}------", end_time_3 - start_time_3)
    
        # assert torch.allclose(native_output, triton_output, 1e-5, 1e-8)

run_compare()

