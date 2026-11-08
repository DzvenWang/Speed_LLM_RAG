"""Direct (no Retrieval at inference) residual + counterfactual forecasting.

Full reference and TRAIN ground truth NEVER appear in forward().
Normal branch uses independent encoder and is mathematically invariant to CF graph.
"""
from __future__ import annotations
import torch
from torch import nn
from .graph_encoder import CFGraphEncoder


def bounded(x,bound):
    return float(bound)*torch.tanh(x.float()/float(bound))


def query_features(x,mask,base):
    """Ten mask-safe per-node features [B,N,10], no access to full data."""
    x=x.float();m=mask.bool()
    B,L,N=x.shape
    coverage=m.float().mean(1)
    recent=m[:,-min(3,L):].float().mean(1)
    first=m.float().argmax(1)
    has=m.any(1)
    last=(L-1)-torch.flip(m,(1,)).float().argmax(1)
    last=torch.where(has,last,torch.zeros_like(last))
    first=torch.where(has,first,torch.zeros_like(first))
    xlast=x.gather(1,last[:,None,:]).squeeze(1)
    xfirst=x.gather(1,first[:,None,:]).squeeze(1)
    xlast=torch.where(has,xlast,torch.zeros_like(xlast))
    xfirst=torch.where(has,xfirst,torch.zeros_like(xfirst))
    mean=torch.where(m,x,torch.zeros_like(x)).sum(1)/m.float().sum(1).clamp_min(1)
    delta=xlast-xfirst
    age=(L-1-last).float()/max(1,L)
    missing_any=(~m).any(1).float()
    all_missing=(~has).float()
    base_mean=base.float().mean(1)
    base_last=base.float()[:,-1]
    return torch.stack([coverage,recent,age,xlast,mean,delta,missing_any,
                        all_missing,base_mean,base_last],dim=-1)


class QueryEncoder(nn.Module):
    def __init__(self,context_dim,hidden=64,horizon=12):
        super().__init__()
        self.horizon=int(horizon)
        self.net=nn.Sequential(nn.Linear(context_dim+10,hidden),nn.GELU(),
                               nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU())
        self.horizon_embeddings=nn.Embedding(horizon,hidden)
    def forward(self,student):
        x=student['x_obs'];m=student['observed'];base=student['base'];c=student['context']
        if x.ndim!=3 or m.shape!=x.shape or base.shape[0]!=len(x) or base.shape[2]!=x.shape[2]:
            raise ValueError('TCRR v9.3 input shape mismatch')
        if base.shape[1]!=self.horizon:raise ValueError('Horizon does not match config')
        if c.shape[:2]!=(len(x),x.shape[-1]):raise ValueError('STLA context node dimension mismatch')
        if not torch.isfinite(base).all() or not torch.isfinite(c).all():
            raise RuntimeError('Non-finite base or context')
        q=self.net(torch.cat([c.float(),query_features(x,m,base)],dim=-1))
        h=self.horizon_embeddings.weight
        return q[:,None]+h[None,:,None]


class ResidualHead(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,z):return self.net(z).squeeze(-1)


class CFResponseHead(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(hidden*2+10,hidden),nn.GELU(),
                                   nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU())
        self.response=nn.Linear(hidden,1);self.logvar=nn.Linear(hidden,1)
        self.damage=nn.Linear(hidden,1)
        nn.init.zeros_(self.response.weight);nn.init.zeros_(self.response.bias)
    def forward(self,z,graph,features):
        H=z.shape[1]
        features=features[:,None].expand(-1,H,-1,-1)
        h=self.encoder(torch.cat((z,graph,features),dim=-1))
        return dict(cf_response=self.response(h).squeeze(-1),
                    cf_logvar=self.logvar(h).squeeze(-1).clamp(-8,6),
                    damage_prediction=self.damage(h).squeeze(-1),
                    cf_features=h)


class CFController(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.utility=nn.Sequential(nn.Linear(hidden+4,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.gate=nn.Sequential(nn.Linear(hidden+5,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,-1.5)
    def forward(self,h,residual,cf,cf_logvar):
        r=residual.float();d=cf.float()
        stats=torch.stack((r,d,r*d,(r-d).abs()),-1)
        uf=torch.cat((h.float(),stats),-1)
        utility=self.utility(uf).squeeze(-1)
        gf=torch.cat((uf,cf_logvar.float()[...,None]),-1)
        # Utility output directly influences decision, not merely an aux loss.
        gate=torch.sigmoid(self.gate(gf).squeeze(-1)+0.5*utility)
        return utility,gate


class TCRRV93(nn.Module):
    def __init__(self,context_dim,num_nodes,horizon=12,hidden=64,bound=3.,
                 mode='incremental',graph_mode='none',road_graph=None,functional=None):
        super().__init__()
        if mode not in ('incremental','decomposed','single'):
            raise ValueError('Unknown v93 mode')
        self.mode=mode;self.horizon=int(horizon);self.bound=float(bound)
        self.normal_query=QueryEncoder(context_dim,hidden,horizon)
        self.residual_head=ResidualHead(hidden)
        # CF must NEVER own or update Normal Query/Residual parameters.
        # Trainable full temporal-mask pattern embedding, independent from frozen Normal.
        self.cf_mask_encoder=nn.Sequential(nn.Linear(12,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.cf_adapter=nn.Sequential(nn.Linear(hidden*2+10,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.graph=CFGraphEncoder(graph_mode,num_nodes,hidden,road_graph,functional)
        self.cf_response_head=CFResponseHead(hidden)
        self.cf_controller=CFController(hidden)
        self.cf_enabled=(mode != 'single')
    def normal(self,student):
        z=self.normal_query(student)
        r=self.residual_head(z)
        return r,z
    def forward(self,student,enable_cf=True):
        if set(student)!={'x_obs','observed','tod','dow','base','context'}:
            raise ValueError('Student forward accepts only visible query features; no teacher targets')
        base=student['base'].float()
        r,z=self.normal(student)
        normal_pred=base+bounded(r,self.bound)
        features=query_features(student['x_obs'],student['observed'],base)
        mask_pattern=self.cf_mask_encoder(student['observed'].float().transpose(1,2))
        mask_context=mask_pattern[:,None].expand(-1,self.horizon,-1,-1)
        cf_state=self.cf_adapter(torch.cat((z.detach(),features[:,None].expand(-1,self.horizon,-1,-1),mask_context),dim=-1))
        graph=self.graph(cf_state,student['observed'])
        cf=self.cf_response_head(cf_state,graph,features)
        d=cf['cf_response'].float()
        u,g=self.cf_controller(cf['cf_features'],r.detach(),d,cf['cf_logvar'])
        if not enable_cf or not self.cf_enabled:g=torch.zeros_like(g)
        # No extra observation failure when truly all input slots are observed.
        no_failure=student['observed'].bool().all(dim=(1,2))
        if no_failure.any():g=torch.where(no_failure[:,None,None],torch.zeros_like(g),g)
        increment=bounded(r.float()+g*d,self.bound)-bounded(r.float(),self.bound)
        prediction=normal_pred+increment
        return dict(base=base,normal_raw=r,normal_prediction=normal_pred,
                    prediction=prediction,cf_response=d,cf_logvar=cf['cf_logvar'],
                    cf_features=cf['cf_features'],damage_prediction=cf['damage_prediction'],
                    cf_utility=u,cf_gate=g,cf_increment=increment)
