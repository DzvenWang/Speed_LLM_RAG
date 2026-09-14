from __future__ import annotations
import torch
from torch import nn


class SensingEncoder(nn.Module):
    def __init__(self, din, hidden=192, demb=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, demb), nn.LayerNorm(demb),
        )
    def forward(self, x): return self.net(x)


class LowRankResidualHead(nn.Module):
    def __init__(self, demb, P, N, rank=16):
        super().__init__(); self.P,self.N,self.rank=P,N,rank
        self.coef=nn.Linear(demb,P*rank); self.node=nn.Parameter(torch.randn(rank,N)*0.02); self.bias=nn.Parameter(torch.zeros(P,N))
    def forward(self,z):
        c=self.coef(z).view(-1,self.P,self.rank); return torch.einsum('bpr,rn->bpn',c,self.node)+self.bias


class M1Model(nn.Module):
    def __init__(self,din,hidden,demb,dropout,P,N,rank):
        super().__init__(); self.encoder=SensingEncoder(din,hidden,demb,dropout); self.head=LowRankResidualHead(demb,P,N,rank)
    def forward(self,x):
        z=self.encoder(x); return self.head(z),z


def _pair_features(q,c,sim): return torch.cat([q,c,torch.abs(q-c),sim[...,None]],-1)


class HorizonUtilityNet(nn.Module):
    """Candidate utility conditioned on explicit horizon-local residual descriptors."""
    def __init__(self,demb,P,local_dim=8,horizon_dim=12):
        super().__init__(); self.P=int(P); self.local_dim=int(local_dim); self.horizon_dim=int(horizon_dim)
        d=3*demb+1
        self.global_body=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,128),nn.GELU())
        self.h_emb=nn.Embedding(self.P,self.horizon_dim)
        self.role=nn.Sequential(nn.Linear(128+self.horizon_dim+self.local_dim,96),nn.GELU(),nn.Dropout(0.05),nn.Linear(96,1))
    def forward(self,q,c,sim,local):
        # q/c [B,K,D], local [B,K,P,L]
        g=self.global_body(_pair_features(q,c,sim)); B,K,_=g.shape
        ge=g[:,:,None,:].expand(B,K,self.P,-1); he=self.h_emb.weight[None,None,:,:].expand(B,K,-1,-1)
        return self.role(torch.cat([ge,he,local],-1)).squeeze(-1)


class RegionHorizonUtilityNet(nn.Module):
    """Candidate utility for graph-region × horizon support roles."""
    def __init__(self,demb,P,R,local_dim=8,horizon_dim=8,region_dim=16):
        super().__init__(); self.P,self.R=int(P),int(R); self.local_dim=int(local_dim); self.horizon_dim=int(horizon_dim); self.region_dim=int(region_dim)
        d=3*demb+1
        self.global_body=nn.Sequential(nn.Linear(d,288),nn.GELU(),nn.Dropout(0.1),nn.Linear(288,128),nn.GELU())
        self.h_emb=nn.Embedding(self.P,self.horizon_dim); self.r_emb=nn.Embedding(self.R,self.region_dim)
        self.role=nn.Sequential(nn.Linear(128+self.horizon_dim+self.region_dim+self.local_dim,112),nn.GELU(),nn.Dropout(0.05),nn.Linear(112,1))
    def forward(self,q,c,sim,local):
        # local [B,K,P,R,L]
        g=self.global_body(_pair_features(q,c,sim)); B,K,_=g.shape
        ge=g[:,:,None,None,:].expand(B,K,self.P,self.R,-1)
        he=self.h_emb.weight[None,None,:,None,:].expand(B,K,self.P,self.R,-1)
        re=self.r_emb.weight[None,None,None,:,:].expand(B,K,self.P,self.R,-1)
        return self.role(torch.cat([ge,he,re,local],-1)).squeeze(-1)


class LocalSafeGate(nn.Module):
    """Shared local gate conditioned on query, node, horizon and sensing state."""
    def __init__(self,demb,P,N,node_dim=16,horizon_dim=8,hidden=96):
        super().__init__(); self.P,self.N=int(P),int(N)
        self.node_emb=nn.Embedding(self.N,node_dim); self.h_emb=nn.Embedding(self.P,horizon_dim)
        din=demb+node_dim+horizon_dim+7
        self.net=nn.Sequential(nn.Linear(din,hidden),nn.GELU(),nn.Dropout(0.1),nn.Linear(hidden,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias,-2.0)
    def forward(self,q,base,corr,obs_fraction,direct,support):
        B,P,N=base.shape
        if P!=self.P or N!=self.N: raise ValueError(f'Gate shape mismatch {(P,N)} vs {(self.P,self.N)}')
        qe=q[:,None,None,:].expand(B,P,N,-1); ne=self.node_emb.weight[None,None,:,:].expand(B,P,N,-1); he=self.h_emb.weight[None,:,None,:].expand(B,P,N,-1)
        obs=obs_fraction[:,None,:,None].expand(B,P,N,1); dm=direct[:,None,:,None].float().expand(B,P,N,1); sm=support[:,None,:,None].float().expand(B,P,N,1)
        local=torch.stack([base,corr,torch.abs(corr),base.abs()],-1)
        return torch.sigmoid(self.net(torch.cat([qe,ne,he,local,obs,dm,sm],-1)).squeeze(-1))
