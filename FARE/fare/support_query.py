"""Forecast-conditioned, mask-aware spatial/temporal query and memory keys.

A region candidate is recalled once; nodes independently score that candidate.
No forecast targets or raw train labels are inputs to the encoders.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ForecastQueryEncoder(nn.Module):
    def __init__(self, history: int, backbone_dim: int, hidden: int = 64, key_dim: int = 128):
        super().__init__()
        self.history = history
        # Per-anchor history preserves spatial identity until AFTER encoding.
        self.temporal = nn.Sequential(nn.Linear(2 * history + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.region_head = nn.Sequential(nn.Linear(hidden * 3 + 3, hidden), nn.GELU(), nn.Linear(hidden, key_dim))
        self.node_head = nn.Sequential(nn.Linear(backbone_dim + hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, key_dim))
        self.key_head = nn.Sequential(nn.Linear(hidden * 3 + 3, hidden), nn.GELU(), nn.Linear(hidden, key_dim))
        self.node_context = nn.Linear(backbone_dim, hidden)

    def _tokens(self, values, valid, regions):
        # [B,G,H,A] -> [B,G,A,D]; preserve anchor direction and ordered trajectory.
        b, g, h, a = values.shape
        x = values.permute(0,1,3,2)
        m = valid.permute(0,1,3,2).float()
        direction = torch.stack([
            regions.up_anchor, regions.down_anchor,
            regions.anchor_valid.float()
        ], -1)[None].expand(b,-1,-1,-1)
        feat = torch.cat([x * m, m, direction], -1)
        return self.temporal(feat)

    def _region_features(self, values, valid, regions):
        tokens = self._tokens(values, valid, regions)
        b, g, a, d = tokens.shape
        coverage = valid.float().mean((2,3))
        weights = valid.any(2).float() * regions.anchor_valid[None].float()
        weights = weights / weights.sum(-1,keepdim=True).clamp_min(1)
        avg = (tokens * weights[...,None]).sum(2)
        up = regions.up_anchor[None] * weights
        down = regions.down_anchor[None] * weights
        up = (tokens * up[...,None]).sum(2) / up.sum(-1,keepdim=True).clamp_min(1e-6)
        down = (tokens * down[...,None]).sum(2) / down.sum(-1,keepdim=True).clamp_min(1e-6)
        feats = torch.cat([avg, up, down,
                           coverage[...,None],
                           (valid.float()*regions.up_anchor[None,:,None,:]).mean((2,3))[...,None],
                           (valid.float()*regions.down_anchor[None,:,None,:]).mean((2,3))[...,None]],-1)
        return feats,tokens

    def forward(self, ctx, regions, base_hidden, base_forecast, x_obs, mask):
        feat,tokens = self._region_features(ctx['anchor_values'],ctx['anchor_mask'],regions)
        region = F.normalize(self.region_head(feat),dim=-1)
        # Node-specific graph weighting retains different upstream/downstream anchors.
        owner = regions.owner
        transfer = regions.graph_transfer[owner,torch.arange(regions.n,device=owner.device)]
        local = (tokens[:,owner] * transfer[None,:,:,None]).sum(2)
        hidden = base_hidden.mean(1)
        observed = mask.float().mean(1)
        trend = (x_obs[:,-1]-x_obs[:,0])*mask[:,-1].float()*mask[:,0].float()
        signal = torch.stack([observed,trend,base_forecast.std(1,unbiased=False)],-1)
        node = F.normalize(self.node_head(torch.cat([hidden,local,signal],-1)),dim=-1)
        return {'region_query':region,'node_query':node,'anchor_tokens':tokens,'node_coverage':observed}

    @torch.no_grad()
    def memory_keys(self, memory, regions, batch=128):
        ids=regions.anchor_ids.flatten();pieces=[]
        for lo in range(0,len(memory.x),batch):
            x=memory.x[lo:lo+batch];v=memory.valid[lo:lo+batch]
            b,h,n=x.shape
            xx=x[:,:,ids].reshape(b,h,regions.g,regions.a).permute(0,2,1,3)
            vv=v[:,:,ids].reshape(b,h,regions.g,regions.a).permute(0,2,1,3)&regions.anchor_valid[None,:,None,:]
            feat,_=self._region_features(xx,vv,regions)
            pieces.append(F.normalize(self.key_head(feat),dim=-1))
        return torch.cat(pieces,0)

    def encode_key_examples(self, x, valid, regions):
        b,h,n=x.shape; ids=regions.anchor_ids.flatten()
        xx=x[:,:,ids].reshape(b,h,regions.g,regions.a).permute(0,2,1,3)
        vv=valid[:,:,ids].reshape(b,h,regions.g,regions.a).permute(0,2,1,3)&regions.anchor_valid[None,:,None,:]
        f,_=self._region_features(xx,vv,regions)
        return F.normalize(self.key_head(f),dim=-1)
