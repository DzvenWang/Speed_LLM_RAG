from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .models import NodeSensingEncoder


class PredictiveNeedNet(nn.Module):
    """Predict the future PatchSTG error state from deployment-safe query context.

    Target during training only:
        E* = Y - Y_base
    Inference input:
        sensing-conditioned node features + frozen PatchSTG forecast already embedded
        inside those node features. No current-query future target is an input.
    """
    def __init__(self, din, N, P, base_start, hidden=192, demb=96,
                 node_dim=24, horizon_dim=16, dropout=0.1):
        super().__init__()
        self.P=int(P); self.base_start=int(base_start)
        self.encoder=NodeSensingEncoder(int(din),int(N),int(hidden),int(demb),int(node_dim),float(dropout))
        self.h_emb=nn.Embedding(self.P,int(horizon_dim))
        self.head=nn.Sequential(
            nn.Linear(int(demb)+int(horizon_dim)+1,int(hidden)),
            nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(hidden),96),nn.GELU(),
            nn.Linear(96,1),
        )

    def forward(self,x):
        # x [B,N,F]; output [B,P,N] in normalized-speed residual units.
        B,N,_=x.shape
        z=self.encoder(x)
        h=self.h_emb.weight[None,None,:,:].expand(B,N,-1,-1)
        ze=z[:,:,None,:].expand(-1,-1,self.P,-1)
        base=x[:,:,self.base_start:self.base_start+self.P]
        if base.shape[-1] != self.P:
            raise RuntimeError(f'PredictiveNeedNet cannot recover base slice from {tuple(x.shape)} start={self.base_start} P={self.P}')
        o=self.head(torch.cat([ze,h,base[:,:,:,None]],-1)).squeeze(-1)
        return o.permute(0,2,1).contiguous()


class SupportUtilityProbeNet(nn.Module):
    """Probe whether Top-K support utility is predictable from deployment-safe state.

    Modes A/B/C all use the same architecture and input dimensionality:
      A: need input is zero (query+support only)
      B: need input is predicted by PredictiveNeedNet (deployable)
      C: need input is true future base error (privileged diagnostic only)
    """
    def __init__(self,node_feature_dim,P,sense_dim=6,context_dim=96,
                 residual_dim=48,pair_dim=128,hidden=192,horizon_dim=16,
                 attention_heads=4,dropout=0.1):
        super().__init__()
        self.P=int(P); self.F=int(node_feature_dim); self.context_dim=int(context_dim); self.pair_dim=int(pair_dim)
        self.qctx=nn.Sequential(nn.LayerNorm(self.F),nn.Linear(self.F,self.context_dim),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(self.context_dim,self.context_dim),nn.GELU())
        self.sctx=nn.Sequential(nn.LayerNorm(self.F),nn.Linear(self.F,self.context_dim),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(self.context_dim,self.context_dim),nn.GELU())
        self.residual=nn.Sequential(nn.LayerNorm(2*self.P),nn.Linear(2*self.P,int(residual_dim)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(residual_dim),int(residual_dim)),nn.GELU())
        pair_in=4*self.context_dim + int(residual_dim) + 3*int(sense_dim) + 1
        self.pair=nn.Sequential(nn.Linear(pair_in,int(hidden)),nn.LayerNorm(int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),self.pair_dim),nn.GELU())
        heads=max(1,int(attention_heads))
        if self.pair_dim % heads != 0: heads=1
        self.cross=nn.MultiheadAttention(self.pair_dim,heads,dropout=float(dropout),batch_first=True)
        self.cross_norm=nn.LayerNorm(self.pair_dim)
        self.h_emb=nn.Embedding(self.P,int(horizon_dim))
        # residual, |residual|, qbase, sbase, sbase-qbase, conf,
        # need, |need|, need*residual -> 9 local scalars.
        role_in=self.pair_dim+int(horizon_dim)+9
        self.utility=nn.Sequential(nn.Linear(role_in,int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),96),nn.GELU(),nn.Linear(96,1))

    def forward(self,qctx,sctx,qs,ss,rr_norm,qbase_norm,sbase_norm,conf,need_norm):
        # qctx [B,N,F], sctx [B,N,L,F], rr [B,N,L,P], need [B,P,N]
        B,N,L,_=sctx.shape
        qx=self.qctx(qctx); sx=self.sctx(sctx)
        qxe=qx[:,:,None,:].expand(-1,-1,L,-1)
        qse=qs[:,:,None,:].expand(-1,-1,L,-1)
        rseq=self.residual(torch.cat([rr_norm,torch.abs(rr_norm)],-1))
        pin=torch.cat([qxe,sx,torch.abs(qxe-sx),qxe*sx,qse,ss,torch.abs(qse-ss),rseq,conf[:,:,:,None]],-1)
        p=self.pair(pin)
        flat=p.reshape(B*N,L,self.pair_dim)
        att,_=self.cross(flat,flat,flat,need_weights=False)
        p=self.cross_norm(flat+att).reshape(B,N,L,self.pair_dim)
        pe=p[:,:,:,None,:].expand(-1,-1,-1,self.P,-1)
        he=self.h_emb.weight[None,None,None,:,:].expand(B,N,L,-1,-1)
        qb=qbase_norm.permute(0,2,1)[:,:,None,:].expand(-1,-1,L,-1)
        need=need_norm.permute(0,2,1)[:,:,None,:].expand(-1,-1,L,-1)
        local=torch.stack([rr_norm,torch.abs(rr_norm),qb,sbase_norm,sbase_norm-qb,conf[:,:,:,None].expand(-1,-1,-1,self.P),need,torch.abs(need),need*rr_norm],-1)
        return self.utility(torch.cat([pe,he,local],-1)).squeeze(-1)


def exact_spearman_sample(x,y,max_points=200000,seed=10):
    x=np.asarray(x,np.float64).reshape(-1); y=np.asarray(y,np.float64).reshape(-1)
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<3: return None
    if len(x)>int(max_points):
        rng=np.random.default_rng(seed); ids=rng.choice(len(x),size=int(max_points),replace=False); x=x[ids]; y=y[ids]
    # Average-tie correction is unnecessary for these mostly-continuous probe outputs;
    # stable double argsort gives a deterministic rank diagnostic without scipy dependency.
    rx=np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort').astype(np.float64)
    ry=np.argsort(np.argsort(y,kind='mergesort'),kind='mergesort').astype(np.float64)
    if rx.std()<1e-12 or ry.std()<1e-12: return 0.0
    return float(np.corrcoef(rx,ry)[0,1])


def pearson(x,y):
    x=np.asarray(x,np.float64).reshape(-1); y=np.asarray(y,np.float64).reshape(-1)
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    if len(x)<3 or x.std()<1e-12 or y.std()<1e-12: return 0.0
    return float(np.corrcoef(x,y)[0,1])


def binary_auroc(scores,labels,max_points=500000,seed=10):
    s=np.asarray(scores,np.float64).reshape(-1); y=np.asarray(labels,bool).reshape(-1)
    m=np.isfinite(s); s=s[m]; y=y[m]
    if len(s)>int(max_points):
        rng=np.random.default_rng(seed); ids=rng.choice(len(s),size=int(max_points),replace=False); s=s[ids]; y=y[ids]
    p=int(y.sum()); n=int((~y).sum())
    if p==0 or n==0: return None
    order=np.argsort(s,kind='mergesort'); ranks=np.empty(len(s),np.float64); ranks[order]=np.arange(1,len(s)+1,dtype=np.float64)
    # Mann-Whitney U / (P*N). Ties are rare for learned continuous scores.
    u=float(ranks[y].sum()-p*(p+1)/2.0)
    return float(u/(p*n))
