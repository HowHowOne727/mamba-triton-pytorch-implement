import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.kernel_size: int = kernel_size
        self.conv = nn.Conv1d(d_model , d_model , kernel_size , 1 , kernel_size - 1 , groups=d_model , bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B , L , D)
        x = x.transpose(1 , 2)  # (B , D , L)
        x = self.conv.forward(x)[: , : , :-(self.kernel_size-1)].transpose(1 , 2)   # (B , L , D)
        return x.contiguous()