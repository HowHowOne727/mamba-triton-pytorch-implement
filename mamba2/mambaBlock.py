import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

from causalConv1d import CausalConv1d


@dataclass
class Mamba2Config:
    num_heads: int
    dim_model: int
    dim_state: int
    expand_factor: float
    conv_size: int



class Mamba2Block(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.n_heads = config.num_heads
        self.outer_d_model = config.dim_model
        self.d_model = int(config.dim_model * config.expand_factor)
        self.d_state = config.dim_state
        self.head_dim = self.d_model // self.n_heads
        self.BC_dim = self.d_state * self.n_heads
        assert self.d_model % self.n_heads == 0
        
        self.A_param = nn.Parameter(torch.zeros(size=[self.n_heads], dtype=torch.float32))
        self.in_proj = nn.Linear(self.outer_d_model, 2 * self.d_model + 2 * self.BC_dim + self.n_heads)      # x, B, C, delta, and skip gate projection
        self.conv = CausalConv1d(self.d_model + 2 * self.BC_dim + self.n_heads, config.conv_size)        # x, B, C, delta conv
        self.out_proj = nn.Linear(self.d_model, self.outer_d_model)
    
    def forward(self, x: Tensor):
        Batch, SeqLen, _ = x.shape
        proj = self.in_proj.forward(x)
        u, v = torch.split(proj, [self.d_model + 2 * self.BC_dim + self.n_heads, self.d_model], dim=-1)
        u = self.conv.forward(u)
        X, B, C, delta = torch.split(u, [self.d_model, self.BC_dim, self.BC_dim, self.n_heads], dim=-1)
        X = X.view(Batch, SeqLen, self .n_heads, self.head_dim)
        B = B.view(Batch, SeqLen, self.n_heads, self.d_state)
        C = C.view(Batch, SeqLen, self.n_heads, self.d_state)
        # WIP...