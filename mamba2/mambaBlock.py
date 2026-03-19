import torch
import torch.nn as nn
from torch import Tensor

from .ssm_kernel import mamba2_fwd
from causalConv1d import CausalConv1d



class Mamba2Layer(nn.Module):
    def __init__(
        self,
        outer_dim: int,
        inner_dim: int,
        state_dim: int,
        n_heads: int,
        conv_kernel_size: int
    ) -> None:
        super().__init__()
        self.outer_dim = outer_dim
        self.inner_dim = inner_dim
        self.d_state = state_dim
        self.n_heads = n_heads

        assert inner_dim % n_heads == 0, "inner_dim can't be divided by n_heads"
        self.d_head = inner_dim // n_heads

        proj_dim = 2 * n_heads + inner_dim + 2 * self.d_state   # A, delta, x, B, C
        self.in_proj = nn.Linear(self.outer_dim, proj_dim)
        self.conv = CausalConv1d(proj_dim, conv_kernel_size)
        self.out_proj = nn.Linear(self.inner_dim, self.outer_dim)

        nn.init.normal_(self.in_proj.weight, 0, 0.02)
    
    def forward(self, input: Tensor, h0: Tensor|None = None):
        batchs = input.size(0)
        if h0 is None:
            h0 = torch.zeros(size=(batchs, self.n_heads, self.d_state, self.d_head), dtype=torch.float32, device=input.device)
        
        input = self.in_proj(input)
        input = self.conv(input)
        A, delta, x, B, C = torch.split(input, [self.n_heads, self.n_heads, self.inner_dim, self.d_state, self.d_state], dim=-1)
        y, hn = mamba2_fwd(A, delta, x, B, C, h0, input.device != "cuda")
        y = y.view(batchs, input.size(1), self.inner_dim)
        return y, hn


