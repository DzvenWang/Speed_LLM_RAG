from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SpatioTemporalEncoder


class ContextRestorer(nn.Module):
    """Dense endogenous restoration + forecast intervention control.

    The raw estimate exists for every artificial-missing cell.  A separate
    control gate decides how strongly that estimate should replace the frozen
    Backbone's native incomplete-input representation.
    """
    def __init__(self, num_nodes: int, adj, hidden_dim=128, num_layers=4, num_heads=8,
                 dropout=0.1, ff_mult=4, gate_init_bias=-3.0):
        super().__init__()
        self.encoder = SpatioTemporalEncoder(num_nodes,adj,hidden_dim,num_layers,num_heads,dropout,ff_mult)
        self.readout = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1))
        self.control = nn.Sequential(
            nn.Linear(hidden_dim+4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Linear(hidden_dim//2,1),
        )
        nn.init.zeros_(self.control[-1].weight)
        nn.init.constant_(self.control[-1].bias, float(gate_init_bias))

    def encode(self, x, observed, tod, dow):
        return self.encoder(x,observed,tod,dow)

    def global_embedding(self, x, observed, tod, dow):
        h = self.encode(x,observed,tod,dow)
        return self.encoder.global_embedding(h, observed)

    def forward(self, x, observed, artificial, tod, dow):
        h = self.encode(x,observed,tod,dow)
        estimate = self.readout(h).squeeze(-1)
        node_obs = observed.float().mean(1)[:,None,:].expand_as(x)
        query_obs = observed.float().mean((1,2))[:,None,None].expand_as(x)
        delta = estimate - x
        feat = torch.cat([h, x[...,None], delta[...,None], node_obs[...,None], query_obs[...,None]], dim=-1)
        gate = torch.sigmoid(self.control(feat).squeeze(-1))
        controlled_estimate = x + gate * delta
        restored = torch.where(artificial.bool(), controlled_estimate, x)
        raw_restored = torch.where(artificial.bool(), estimate, x)
        return restored, {"estimate":estimate, "raw_restored":raw_restored, "gate":gate}
