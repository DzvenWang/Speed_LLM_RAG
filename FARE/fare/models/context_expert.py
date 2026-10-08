from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SpatioTemporalEncoder
from .intervention import HierarchicalInterventionController


class ContextRestorer(nn.Module):
    """Dense endogenous restorer with hierarchical forecast intervention control."""
    def __init__(self, num_nodes: int, adj, hidden_dim=128, num_layers=4, num_heads=8,
                 dropout=0.1, ff_mult=4, alpha_grid=(0., .25, .5, .75, 1.0)):
        super().__init__()
        self.encoder = SpatioTemporalEncoder(num_nodes,adj,hidden_dim,num_layers,num_heads,dropout,ff_mult)
        self.readout = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1))
        # h + x + delta + observed + artificial
        self.intervention = HierarchicalInterventionController(hidden_dim + 4, alpha_grid, hidden_dim, dropout)

    def encode(self, x, observed, tod, dow):
        return self.encoder(x,observed,tod,dow)

    def global_embedding(self, x, observed, tod, dow):
        h = self.encode(x,observed,tod,dow)
        return self.encoder.global_embedding(h, observed)

    def forward(self, x, observed, artificial, tod, dow, alpha_override=None):
        h = self.encode(x,observed,tod,dow)
        estimate = self.readout(h).squeeze(-1)
        delta = estimate - x
        cell = torch.cat([
            h, x[...,None], delta[...,None], observed.float()[...,None], artificial.float()[...,None]
        ], dim=-1)
        ctrl = self.intervention(cell,artificial,observed,delta,alpha_override)
        controlled_estimate = x + ctrl['gate'] * delta
        restored = torch.where(artificial.bool(), controlled_estimate, x)
        raw_restored = torch.where(artificial.bool(), estimate, x)
        return restored, {"estimate":estimate, "raw_restored":raw_restored, **ctrl}
