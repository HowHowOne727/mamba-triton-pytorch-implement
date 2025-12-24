import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba
from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int
    expansion_factor: float
    n_states: int
    dt_rank: int
    n_layers: int
    vocab_size: int
    conv_kernel_size: int

    padding_idx: int


class MambaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.padding_idx)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        self.output_proj.weight = self.embed.weight
        self.out_norm = nn.LayerNorm(config.d_model)

        self.mamba = Mamba(config.d_model, config.expansion_factor, config.n_states, config.dt_rank, config.n_layers, config.conv_kernel_size)
        
        nn.init.normal_(self.output_proj.weight, 0, 0.02)
    def forward(self, ids: torch.Tensor):
        emb = self.embed.forward(ids)

        emb = self.mamba.forward(emb)       # mamba forward
        emb = self.out_norm.forward(emb)    # norm
        logit = self.output_proj.forward(emb)
        return logit