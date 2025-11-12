# Licensed under the Apache License, Version 2.0 (the "License");
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
import torch_npu
from ._ops import add_op_namespace_prefix


@torch.library.custom_op(add_op_namespace_prefix("RMSNorm"), mutates_args=())
def _RMSNorm(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the RMSNorm layer.
    Args:
        x (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The output tensor after applying RMSNorm.
    """
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]
