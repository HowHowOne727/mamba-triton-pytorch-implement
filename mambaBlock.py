import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from mambaKernel import mamba_fusion_ssm

@dataclass
class ModelConfig:
    d_model: int
    num_B_groups: int
    n_states: int
    conv_kernel: int
    n_layers: int
    vocab_size: int

class CausalConv1d(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, num_group: int):
        super().__init__()
        self.kernel_size: int = kernel_size
        self.conv = nn.Conv1d(d_model , d_model , kernel_size , 1 , kernel_size - 1 , groups=num_group)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B , L , D)
        x = x.transpose(1 , 2)  # (B , D , L)
        x = self.conv.forward(x)[: , : , :-(self.kernel_size-1)].transpose(1 , 2)   # (B , L , D)
        return x.contiguous()

class MambaBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(config.d_model , 2 * config.d_model)

        self.delta_proj = nn.Linear(config.d_model , config.d_model , bias=False)
        
        A_init = torch.arange(1 , config.n_states + 1 , 1 , dtype=torch.float32).repeat(config.d_model , 1).contiguous()
        self.A_log_param = nn.Parameter(torch.log(A_init))
        self.B_proj = nn.Linear(config.d_model , config.num_B_groups * config.n_states)
        self.C_param = nn.Parameter(torch.randn(config.d_model , config.n_states) * 0.02)

        self.conv = CausalConv1d(config.d_model , config.conv_kernel , config.d_model)

        nn.init.normal_(self.in_proj.weight, 0, 0.02)
        nn.init.normal_(self.B_proj.weight , 0 , 0.0001)
        nn.init.uniform_(self.delta_proj.weight, 0.001, 0.1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input : (B , L , D)
        BATCH_SIZE , LENGTH , _ = input.shape
        x_proj = self.in_proj.forward(input)    # (B , L , 2*D)

        u , v = torch.split(x_proj , [self.config.d_model , self.config.d_model] , dim=-1)

        delta = F.softplus(self.delta_proj.forward(u))  # (B , L , D)
        A = -torch.exp(self.A_log_param)
        B = self.B_proj.forward(u).view(BATCH_SIZE , LENGTH , self.config.num_B_groups , self.config.n_states)
        u = F.silu(self.conv.forward(u))
        
        y , h_out = mamba_fusion_ssm(delta , A , B , self.C_param , u)

        out = y * F.silu(v)     # (B , L , D)
        return out
    
class Mamba(nn.Module):
    def __init__(self , config: ModelConfig):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size , config.d_model)
        self.output_proj = nn.Linear(config.d_model , config.vocab_size)
        self.output_proj.weight = self.embed.weight

        nn.init.normal_(self.output_proj.weight, 0, 0.02)

        _layers: list[MambaBlock] = []
        _rms_norms: list[nn.RMSNorm] = []
        for _ in range(config.n_layers):
            _layers.append(MambaBlock(config))
            _rms_norms.append(nn.RMSNorm(config.d_model))
        self._layers = _layers
        self.layers = nn.ModuleList(_layers)
        self.rms_norms = nn.ModuleList(_rms_norms)

    def forward(self , ids: torch.Tensor):
        emb = self.embed.forward(ids)

        for i in range(len(self.layers)):
            emb = self.rms_norms[i].forward(self.layers[i].forward(emb) + emb)
        
        logit = self.output_proj.forward(emb)
        return logit