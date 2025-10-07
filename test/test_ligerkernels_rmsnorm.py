import torch
import torch.nn as nn
from tritonrms.rms_norm import LigerRMSNormFunction
import sys
import time
import matplotlib.pyplot as plt


batch, seq_length = 128, 1024
hidden_size = range(128, 5600, 128)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).cuda()
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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
    X_ori = []
    Y_1 = []
    Y_2 = []
    scaleout = 1000
    
    for i in range(len(hidden_size)):
        X = torch.randn(batch, seq_length, hidden_size[i]).cuda()
        
        native_rmsnorm = Qwen3RMSNorm(hidden_size[i])
        start_time_1 = time.time()
        native_output = native_rmsnorm(X)
        end_time_1 = time.time()
        Y_1.append((end_time_1 - start_time_1) * scaleout)
        # print(f"native_output of hidden_size={hidden_size[i]}--------", native_output)
        print(f"native_time of hidden_size={hidden_size[i]}---", end_time_1 - start_time_1)
        

        triton_rmsnorm = LigerRMSNorm(hidden_size[i])
        start_time_2 = time.time()
        triton_output = triton_rmsnorm(X)
        end_time_2 = time.time()
        Y_2.append((end_time_2 - start_time_2) * scaleout)
        X_ori.append(hidden_size[i])
        # print(f"triton_output of hidden_size={hidden_size[i]}--------", triton_output)
        print(f"triton_time of hidden_size={hidden_size[i]}---", end_time_2 - start_time_2)
    
        assert torch.allclose(native_output, triton_output, 1e-5, 1e-8)

    return X_ori, Y_1, Y_2


X_ori, Y_1, Y_2 = run_compare()
plt.xticks(hidden_size)
plt.plot(X_ori[5:], Y_1[5:], label='native', color="green")
plt.plot(X_ori[5:], Y_2[5:], label='triton', color="red")
plt.savefig('result.jpg')
