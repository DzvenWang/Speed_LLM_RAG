from __future__ import annotations
import torch
from torch import nn


class P1ResidualAdapter(nn.Module):
    """Small shared node-wise residual adapter with strict zero-start output.

    Every variant has identical parameters. Controls differ only in whether support
    channels contain zero, causality-preserving random evidence, history evidence,
    or history+subsequent-evolution evidence.
    """
    def __init__(self, latent_dim: int, history_dim: int, horizon: int,
                 hidden: int = 128, dropout: float = 0.10):
        super().__init__()
        self.latent_dim=int(latent_dim); self.history_dim=int(history_dim); self.horizon=int(horizon)
        # query latent + query history + base forecast + support history + support future + future coverage
        inp=self.latent_dim + self.history_dim + self.horizon + self.history_dim + self.horizon + self.horizon
        self.net=nn.Sequential(
            nn.LayerNorm(inp),
            nn.Linear(inp,int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden),int(hidden)//2), nn.GELU(),
            nn.Linear(int(hidden)//2,self.horizon),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_pred, query_latent, query_hist, support_hist, support_future, support_future_cov):
        # base_pred [B,P,N], latent/history [B,N,*], support_future/cov [B,P,N]
        base_node=base_pred.transpose(1,2)
        sf=support_future.transpose(1,2); sc=support_future_cov.transpose(1,2)
        x=torch.cat([query_latent,query_hist,base_node,support_hist,sf,sc],dim=-1)
        delta=self.net(x).transpose(1,2)
        return base_pred + delta, delta
