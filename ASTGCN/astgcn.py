from cProfile import run
from dataclasses import dataclass
from unicodedata import name

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from ASTGCN.blocks.STblock import STBlock

@dataclass
class params:

    # Parameters for data
    in_features: int = 3
    out_features: int = 3

    n_timesteps: int = 0
    n_features: int = 0
    n_verticies: int = 0

    # Parameters for model architecture
    n_layers: int = 3

    # Parameters for Spatial Attention
    w1_size: tuple = (n_timesteps,)
    w2_size: tuple = (n_features, n_timesteps)
    w3_size: tuple = (n_features,)
    s_bias_size: tuple = (n_verticies, n_verticies)
    vs_size: tuple = (n_verticies, n_verticies)
    polynomial_order: int = 3

    # Parameters for Temporal Attention
    u1_size: tuple = (n_verticies,)
    u2_size: tuple = (n_features, n_verticies)
    u3_size: tuple = (n_features,)
    vt_size: tuple = (n_timesteps, n_timesteps)

    # Parameters for Temporal Conv
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dialation: int = 1
    
    # Parameters for Spatial Conv
    polynomial_order: int = 3
    w_sc_size: tuple = (polynomial_order, in_features, out_features)
    cheb_polynomials: torch.tensor = 0 # size = (polynomial_order, n_verticies, n_verticies))

    # Hyperparameters
    lr: int = 1e-3
    dropout_rate: float = 0.2


class ASTGCN(nn.Module):

    def __init__(self, params: params):

        super(ASTGCN, self).__init__()

        self.params = params

        self.blocks = nn.ModuleDict({

            "recent": nn.ModuleList(STBlock(self.params) for _ in range(self.params.n_layers)),
            "daily_p": nn.ModuleList(STBlock(self.params) for _ in range(self.params.n_layers)),
            "weekly": nn.ModuleList(STBlock(self.params) for _ in range(self.params.n_layers))
        
        })

        self.linear_recent = nn.Linear(params.linear_in, params.linear_out, bias=False)
        self.linear_daily_p = nn.Linear(params.linear_in, params.linear_out, bias=False)
        self.linear_weekly = nn.Linear(params.linear_in, params.linear_out, bias=False)

        self.relu = nn.ReLU()
        
        self.recent.append(self.linear_recent)
        self.daily_p.append(self.linear_daily_p)
        self.weekly.append(self.linear_weekly)

        self.loss = nn.MSELoss()
        self.optimizer = optimizer.AdamW(self.parameters(), lr=params.lr)

    def forward(self, x_recent, x_daily_p, x_weekly):

        for layer_recent, layer_daily_p, layer_weekly in zip(self.blocks["recent"], self.blocks["daily_p"], self.blocks["weekly"]):
            
            x_recent = layer_recent.forward(x_recent)
            x_daily_p = layer_daily_p.forward(x_daily_p)
            x_weekly = layer_weekly.forward(x_weekly)

        x_recent = self.relu(self.linear_recent(x_recent))
        x_daily_p = self.relu(self.linear_daily_p(x_daily_p))
        x_weekly = self.relu(self.linear_weekly(x_weekly))

        x = x_recent + x_daily_p + x_weekly 
        
        return F.gelu(x)

    def compute_loss(self, y_pred, y_true):

        return self.loss(y_pred, y_true)


