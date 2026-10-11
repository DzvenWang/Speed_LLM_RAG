"""Forecast-conditioned support demand derived solely from observable inputs."""
import torch
from torch import nn

class SupportDemandEncoder(nn.Module):
    def __init__(self, backbone_dim, query_dim=128, hidden=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(backbone_dim+5,hidden),nn.GELU(),nn.Linear(hidden,query_dim))
        nn.init.normal_(self.net[-1].weight,std=0.002)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,hidden,mask,base,regions):
        # [B,H,N,D] pooled only over observed input; no ground truth used.
        h=hidden.mean(1)
        cov=mask.float().mean(1)
        rough=base.detach().std(dim=1,unbiased=False)
        f=torch.cat([h,cov[...,None],(1-cov)[...,None],rough[...,None],
                     base.detach().mean(1)[...,None],mask.float()[:,-1,:,None]],dim=-1)
        region=torch.einsum('bnd,gn->bgd',f,regions.target_mask)/regions.target_mask.sum(-1)[None,:,None].clamp_min(1)
        return self.net(region)
