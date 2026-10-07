from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SpatioTemporalEncoder


class ContextRestorer(nn.Module):
    """Full spatiotemporal restoration expert.

    Uses only the current incomplete query, mask, graph, TOD/DOW and learned
    parameters. No historical instance memory is read at inference.
    """
    def __init__(self, num_nodes: int, adj, hidden_dim=128, num_layers=4, num_heads=8, dropout=0.1, ff_mult=4):
        super().__init__()
        self.encoder = SpatioTemporalEncoder(num_nodes,adj,hidden_dim,num_layers,num_heads,dropout,ff_mult)
        self.readout = nn.Sequential(nn.Linear(hidden_dim,hidden_dim), nn.GELU(), nn.Linear(hidden_dim,1))

    def encode(self, x, observed, tod, dow):
        return self.encoder(x,observed,tod,dow)

    def global_embedding(self, x, observed, tod, dow):
        h = self.encode(x,observed,tod,dow)
        return self.encoder.global_embedding(h, observed)

    def forward(self, x, observed, artificial, tod, dow):
        h = self.encode(x,observed,tod,dow)
        estimate = self.readout(h).squeeze(-1)
        restored = torch.where(artificial.bool(), estimate, x)
        return restored, h
