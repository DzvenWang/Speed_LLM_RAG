from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import GraphTemporalBlock, normalized_adjacency
from .intervention import HierarchicalInterventionController


class BackboneAwareRestorer(nn.Module):
    """Forecast-conditioned dense restoration + hierarchical intervention landscape."""
    def __init__(self, num_nodes: int, adj, horizon=12, hidden_dim=128, num_layers=4,
                 num_heads=8, dropout=0.1, ff_mult=4,
                 alpha_grid=(0., .25, .5, .75, 1.0)):
        super().__init__()
        self.num_nodes=int(num_nodes); self.horizon=int(horizon)
        self.input_proj = nn.Linear(4, hidden_dim)
        self.forecast_proj = nn.Sequential(nn.Linear(horizon, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.tod_emb = nn.Embedding(288, hidden_dim)
        self.dow_emb = nn.Embedding(7, hidden_dim)
        self.in_ln = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([GraphTemporalBlock(hidden_dim,num_heads,ff_mult,dropout) for _ in range(num_layers)])
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.residual = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
        # h + x + delta + observed + artificial + base uncertainty
        self.intervention = HierarchicalInterventionController(hidden_dim + 5, alpha_grid, hidden_dim, dropout)
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.10))
        self.register_buffer('a_norm', normalized_adjacency(adj), persistent=True)

    def forward(self, x, observed, artificial, tod, dow, context_restored, base_forecast_norm,
                alpha_override=None):
        b,t,n=x.shape
        inp=torch.stack([x,observed.float(),context_restored,context_restored-x],dim=-1)
        h=self.input_proj(inp)
        fp=self.forecast_proj(base_forecast_norm.transpose(1,2))
        h=h+fp[:,None,:,:]
        nodes=torch.arange(n,device=x.device)
        h=h+self.node_emb(nodes)[None,None,:,:]
        h=h+self.tod_emb(tod.long().clamp(0,287))[:,:,None,:]+self.dow_emb(dow.long().clamp(0,6))[:,:,None,:]
        h=self.in_ln(h)
        for blk in self.blocks: h=blk(h,self.a_norm)
        h=self.out_ln(h)
        delta=self.residual(h).squeeze(-1)
        estimate=context_restored + torch.tanh(self.scale) * delta
        restore_delta=estimate-x
        base_uncert=base_forecast_norm.std(1,unbiased=False)[:,None,:].expand_as(x)
        cell=torch.cat([
            h,x[...,None],restore_delta[...,None],observed.float()[...,None],
            artificial.float()[...,None],base_uncert[...,None]
        ],dim=-1)
        ctrl=self.intervention(cell,artificial,observed,restore_delta,alpha_override)
        controlled=x + ctrl['gate']*restore_delta
        restored=torch.where(artificial.bool(),controlled,x)
        raw_restored=torch.where(artificial.bool(),estimate,x)
        return restored, {'delta':delta,'estimate':estimate,'raw_restored':raw_restored,**ctrl}
