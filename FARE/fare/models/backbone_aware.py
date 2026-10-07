from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import GraphTemporalBlock, normalized_adjacency


class BackboneAwareRestorer(nn.Module):
    """Forecast-conditioned restoration expert.

    It consumes the frozen STLAformer forecast produced from the incomplete
    query. This is inference-visible model knowledge, not future ground truth.
    The network learns a residual over the strong Context restoration.
    """
    def __init__(self, num_nodes: int, adj, horizon=12, hidden_dim=128, num_layers=4,
                 num_heads=8, dropout=0.1, ff_mult=4):
        super().__init__()
        self.num_nodes=int(num_nodes); self.horizon=int(horizon)
        self.input_proj = nn.Linear(4, hidden_dim)  # x, observed, context, context-x
        self.forecast_proj = nn.Sequential(nn.Linear(horizon, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.tod_emb = nn.Embedding(288, hidden_dim)
        self.dow_emb = nn.Embedding(7, hidden_dim)
        self.in_ln = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([GraphTemporalBlock(hidden_dim,num_heads,ff_mult,dropout) for _ in range(num_layers)])
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.residual = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.10))
        self.register_buffer('a_norm', normalized_adjacency(adj), persistent=True)

    def forward(self, x, observed, artificial, tod, dow, context_restored, base_forecast_norm):
        b,t,n=x.shape
        inp=torch.stack([x,observed.float(),context_restored,context_restored-x],dim=-1)
        h=self.input_proj(inp)
        fp=self.forecast_proj(base_forecast_norm.transpose(1,2)) # [B,N,H]
        h=h+fp[:,None,:,:]
        nodes=torch.arange(n,device=x.device)
        h=h+self.node_emb(nodes)[None,None,:,:]
        h=h+self.tod_emb(tod.long().clamp(0,287))[:,:,None,:]+self.dow_emb(dow.long().clamp(0,6))[:,:,None,:]
        h=self.in_ln(h)
        for blk in self.blocks: h=blk(h,self.a_norm)
        delta=self.residual(self.out_ln(h)).squeeze(-1)
        estimate=context_restored + torch.tanh(self.scale) * delta
        restored=torch.where(artificial.bool(),estimate,x)
        return restored, delta
