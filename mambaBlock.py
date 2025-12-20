import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm_kernel import mamba_fusion_ssm


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
    def __init__(self, d_model: int, n_states: int, conv_kernel_size: int):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        self.conv_kernel_size = conv_kernel_size

        self.in_proj = nn.Linear(self.d_model , (self.d_model + self.d_model + self.n_states + 1))   # n_states for B, 1 for delta
        
        A_init = torch.arange(1 , self.n_states + 1 , 1 , dtype=torch.float32).repeat(self.d_model , 1).contiguous()
        self.A_log_param = nn.Parameter(torch.log(A_init))
        self.B_param = nn.Parameter(torch.randn(self.n_states) / self.n_states**0.5)    # (N)
        self.C_param = nn.Parameter(torch.randn(self.d_model , self.n_states) / self.n_states**0.5)   # (D , N)

        self.conv = CausalConv1d(self.d_model , self.conv_kernel_size , self.d_model)

        nn.init.normal_(self.in_proj.weight, 0, 0.02)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input : (B , L , D)
        x_proj = self.in_proj.forward(input)

        v , x , B , delta = torch.split(x_proj , [self.d_model , self.d_model , self.n_states , 1] , dim=-1)

        delta = F.softplus(delta).squeeze(-1)  # (B , L , 1) to (B , L)
        delta = delta * F.sigmoid(delta)
        A = -torch.exp(self.A_log_param)
        B = (B + self.B_param[None , None , :])
        x = self.conv.forward(F.silu(x))
        
        y , h_out = mamba_fusion_ssm(delta , A , B , self.C_param , x)
        out = y * F.silu(v)     # (B , L , D)
        return out
    
class Mamba(nn.Module):
    def __init__(self, d_model: int, n_states: int, n_layers: int, conv_kernel_size: int):
        super().__init__()

        _layers: list[MambaBlock] = []
        _rms_norms: list[nn.RMSNorm] = []
        for _ in range(n_layers):
            _rms_norms.append(nn.RMSNorm(d_model))
            _layers.append(MambaBlock(d_model, n_states, conv_kernel_size))
        self._layers = _layers
        self.layers = nn.ModuleList(_layers)
        self.rms_norms = nn.ModuleList(_rms_norms)

    def forward(self , x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.rms_norms[i](x)
            x = self.layers[i].forward(x) + x
        return x