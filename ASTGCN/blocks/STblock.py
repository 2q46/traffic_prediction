
import torch
import torch.nn as nn

from ASTGCN.blocks.attnblock import SpatialAttention, TemporalAttention
from ASTGCN.blocks.convblock import SpatialConv, TemporalConv

class STBlock(nn.Module):

    def __init__(self, params):

        super(STBlock, self).__init__()

        self.params = params

        self.spatial_attn = SpatialAttention(
            params=self.params
        )

        self.temporal_attn = TemporalAttention(
            params=self.params
        )

        self.spatial_conv = SpatialConv(
            params=self.params
        )

        self.temporal_conv = TemporalConv(
            params=self.params
        )

        self.ln = nn.LayerNorm(params.in_features)
        self.dropout = nn.Dropout(params.dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):

        # x (B,N,F,T)
        spatial_attn = self.spatial_attn.forward(x) # (B,N,N)
        temporal_attn = self.temporal_attn.forward(x) # (B,T,T)
        spatial_conv = self.spatial_conv.forward(x, spatial_attn) # (B,N,O,T)
        output = self.temporal_conv.forward(spatial_conv, temporal_attn) # (B,N,F,T)
        
        output = output.permute(0,1,3,2) # swap F and T  ... T, F)
        output = self.ln(output)
        output = output.permute(0,1,3,2) # swap F and T ... F, T)
        output = self.dropout(output)

        return self.gelu(output)
