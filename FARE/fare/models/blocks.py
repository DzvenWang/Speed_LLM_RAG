from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(adj, device=None):
    a = torch.tensor(adj, dtype=torch.float32, device=device)
    a = torch.clamp(a, min=0.0)
    a = a + a.transpose(0, 1)
    a.fill_diagonal_(1.0)
    rs = a.sum(-1, keepdim=True).clamp_min(1e-6)
    return a / rs


class GraphTemporalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.temporal_ln = nn.LayerNorm(hidden)
        self.temporal_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.temporal_drop = nn.Dropout(dropout)
        self.graph_ln = nn.LayerNorm(hidden)
        self.graph_self = nn.Linear(hidden, hidden)
        self.graph_msg = nn.Linear(hidden, hidden)
        self.graph_gate = nn.Linear(hidden * 2, hidden)
        self.graph_drop = nn.Dropout(dropout)
        self.ff_ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * ff_mult, hidden), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        # h [B,T,N,H]
        b,t,n,d = h.shape
        z = self.temporal_ln(h).permute(0,2,1,3).reshape(b*n,t,d)
        attn,_ = self.temporal_attn(z,z,z,need_weights=False)
        h = h + self.temporal_drop(attn.reshape(b,n,t,d).permute(0,2,1,3))

        z = self.graph_ln(h)
        msg = torch.einsum('ij,btjh->btih', a_norm, z)
        gate = torch.sigmoid(self.graph_gate(torch.cat([z,msg],dim=-1)))
        g = gate * self.graph_msg(msg) + (1.0-gate) * self.graph_self(z)
        h = h + self.graph_drop(g)
        h = h + self.ff(self.ff_ln(h))
        return h


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, num_nodes: int, adj, hidden_dim: int=128, num_layers: int=4,
                 num_heads: int=8, dropout: float=0.1, ff_mult: int=4):
        super().__init__()
        self.num_nodes = int(num_nodes); self.hidden_dim=int(hidden_dim)
        self.value_proj = nn.Linear(2, hidden_dim)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.tod_emb = nn.Embedding(288, hidden_dim)
        self.dow_emb = nn.Embedding(7, hidden_dim)
        self.in_ln = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([
            GraphTemporalBlock(hidden_dim,num_heads,ff_mult,dropout) for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.register_buffer('a_norm', normalized_adjacency(adj), persistent=True)

    def forward(self, x: torch.Tensor, observed: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        # x/observed [B,T,N], tod/dow [B,T]
        b,t,n = x.shape
        inp = torch.stack([x, observed.float()], dim=-1)
        h = self.value_proj(inp)
        nodes = torch.arange(n, device=x.device)
        h = h + self.node_emb(nodes)[None,None,:,:]
        h = h + self.tod_emb(tod.long().clamp(0,287))[:,:,None,:]
        h = h + self.dow_emb(dow.long().clamp(0,6))[:,:,None,:]
        h = self.in_ln(h)
        for blk in self.blocks:
            h = blk(h, self.a_norm)
        return self.out_ln(h)

    def global_embedding(self, h: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        # visible-aware mean pooling; hidden already contains mask information.
        w = 0.25 + 0.75 * observed.float()
        z = (h * w[...,None]).sum((1,2)) / w.sum((1,2),keepdim=False).clamp_min(1.0)[:,None]
        return F.normalize(z, dim=-1)
