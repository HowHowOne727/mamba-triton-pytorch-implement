import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm_kernel import mamba_fusion_ssm
import math


class CausalConv1d(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, num_group: int):
        super().__init__()
        self.kernel_size: int = kernel_size
        self.conv = nn.Conv1d(d_model , d_model , kernel_size , 1 , kernel_size - 1 , groups=num_group , bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B , L , D)
        x = x.transpose(1 , 2)  # (B , D , L)
        x = self.conv.forward(x)[: , : , :-(self.kernel_size-1)].transpose(1 , 2)   # (B , L , D)
        return x.contiguous()

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, expansion_factor: float, n_states: int, dt_rank: int, conv_kernel_size: int):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expansion_factor)
        self.n_states = n_states
        self.dt_rank = dt_rank
        self.conv_kernel_size = conv_kernel_size

        self.in_proj = nn.Linear(self.d_model , (self.d_inner + self.d_inner + self.n_states + self.dt_rank))   # n_states for B, 1 for delta
        
        A_init = torch.arange(1 , self.n_states + 1 , 1 , dtype=torch.float32).repeat(self.d_inner , 1).contiguous()
        self.A_log_param = nn.Parameter(torch.log(A_init))
        self.B_param = nn.Parameter(torch.randn(self.n_states) / self.n_states**0.5)    # (N)
        self.C_param = nn.Parameter(torch.randn(self.d_inner , self.n_states) / self.n_states**0.5)   # (D , N)
        self.dt_rank_to_D = nn.Linear(self.dt_rank, self.d_inner)

        _dt_vals = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_rank_to_D.bias = nn.Parameter(torch.log(torch.exp(_dt_vals) - 1))

        self.conv = CausalConv1d(self.d_inner , self.conv_kernel_size , self.d_inner)

        self.output_proj = nn.Linear(self.d_inner, self.d_model)

        nn.init.normal_(self.in_proj.weight, 0, 0.02)
        nn.init.normal_(self.dt_rank_to_D.weight, 0, 0.02)
        nn.init.normal_(self.output_proj.weight, 0, 0.02)
        # nn.init.zeros_(self.dt_rank_to_D.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input : (B , L , D)
        x_proj = self.in_proj.forward(input)

        v , x , B , delta = torch.split(x_proj , [self.d_inner , self.d_inner , self.n_states , self.dt_rank] , dim=-1)

        x = F.silu(self.conv.forward(x))

        with torch.autocast("cuda", enabled=False):
            # cast float32
            delta = delta.float()
            x = x.float()
            B = B.float()

            delta = F.softplus(self.dt_rank_to_D.forward(delta))  # (B , L , dt_rank) -> (B , L , D)
            A = -torch.exp(self.A_log_param.float())
            B = (B + self.B_param.float()[None , None , :])
            
            y , h_out = mamba_fusion_ssm(delta , A , B , self.C_param.float() , x)
            out = y * F.silu(v.float())     # (B , L , D)
        out = out.to(v.dtype)
        return self.output_proj(out)
    
class Mamba(nn.Module):
    def __init__(self, d_model: int, expansion_factor: float, n_states: int, dt_rank: int, n_layers: int, conv_kernel_size: int):
        super().__init__()

        _layers: list[MambaBlock] = []
        _rms_norms: list[nn.RMSNorm] = []
        for _ in range(n_layers):
            _rms_norms.append(nn.RMSNorm(d_model))
            _layers.append(MambaBlock(d_model, expansion_factor, n_states, dt_rank, conv_kernel_size))
        self._layers = _layers
        self.layers = nn.ModuleList(_layers)
        self.rms_norms = nn.ModuleList(_rms_norms)

    def forward(self , x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.rms_norms[i](x)
            x = self.layers[i].forward(x) + x
        return x