"""TCRR v2: local sensing failure representation and horizon-specific error utility.

No current-query future labels, complete query values or test statistics enter this module.
Memory histories are subjected to the *query* observation mask for matching.
"""
from __future__ import annotations
import torch
from torch import nn
from .traffic import node_features


class FailureConditionedTraffic(nn.Module):
    def __init__(self, horizon=12, hidden=48):
        super().__init__()
        self.horizon=int(horizon)
        self.register_buffer('neighbors',torch.empty(0,0,dtype=torch.long),persistent=False)
        self.state=nn.Sequential(nn.Linear(8+6,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.failure=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.regime=nn.Linear(hidden,4)
        self.horizon_embed=nn.Embedding(self.horizon,8)
        # candidate-specific node/horizon utility and normal/failure transfer scales
        self.candidate=nn.Sequential(nn.Linear(3*hidden+8+7,hidden),nn.GELU(),nn.Linear(hidden,5))
        self.mixture=nn.Sequential(nn.Linear(hidden+8+5,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.candidate[-1].weight);nn.init.zeros_(self.candidate[-1].bias)
        nn.init.zeros_(self.mixture[-1].weight);nn.init.zeros_(self.mixture[-1].bias)

    def install_neighbors(self,neighbors):
        if neighbors.ndim!=2:raise ValueError('Neighbor graph must be [N,K]')
        self.neighbors=neighbors.long().to(self.neighbors.device)

    def _node_context(self,x,mask):
        # Features of local alternative observational support, not imputed values.
        m=mask.float()
        v=node_features(x,mask)
        observed=m.mean(1)
        if self.neighbors.numel() and self.neighbors.shape[0]==observed.shape[1]:
            neigh_cov=observed[:,self.neighbors].mean(-1)
        else:
            # Graph is not installed in synthetic tests. Use network-wide observed
            # support, not an unjustified adjacency of sensor array indices.
            neigh_cov=observed.mean(1,keepdim=True).expand_as(observed)
        first=m[:,:max(1,x.shape[1]//3)].mean(1)
        last=m[:,-max(1,x.shape[1]//3):].mean(1)
        age=v[...,5]
        # Eventual actual network graph can replace this approximate local support.
        local=torch.stack([observed,neigh_cov,first,last,age,1-observed],-1)
        return v,local

    def _query(self,x,mask):
        raw,local=self._node_context(x,mask)
        state=self.state(torch.cat([raw,local],-1))
        fail=self.failure(local)
        regime=torch.softmax(self.regime(state),-1)
        entropy=-(regime.clamp_min(1e-7)*regime.clamp_min(1e-7).log()).sum(-1)
        return state,fail,entropy,local

    def forward(self,x,mask,candidate_history,base,normal_residual,failure_residual,disable_failure=False):
        b,k,h,n=candidate_history.shape
        B,P,N=base.shape
        if B!=b or N!=n or P!=self.horizon:raise ValueError('Traffic candidate dimension mismatch')
        q,fail,entropy,support=self._query(x,mask)
        cm=mask[:,None].expand(b,k,h,n).reshape(b*k,h,n)
        ch=candidate_history.reshape(b*k,h,n)
        craw,clocal=self._node_context(ch,cm)
        cz=self.state(torch.cat([craw,clocal],-1)).reshape(b,k,n,-1)
        qz=q[:,None].expand_as(cz)
        mask_condition=fail[:,None].expand_as(cz)
        if disable_failure:mask_condition=torch.zeros_like(mask_condition)
        hh=self.horizon_embed(torch.arange(P,device=x.device))[None,None,:,None,:].expand(b,k,P,n,-1)
        qe=qz[:,:,None].expand(b,k,P,n,-1)
        ce=cz[:,:,None].expand(b,k,P,n,-1)
        fe=mask_condition[:,:,None].expand(b,k,P,n,-1)
        extra=torch.stack([base[:,None].expand(b,k,P,n), normal_residual, failure_residual,
                           normal_residual.abs(),failure_residual.abs(),
                           entropy[:,None,None].expand(b,k,P,n),
                           support[...,0][:,None,None].expand(b,k,P,n)],-1)
        feats=torch.cat([qe,ce,fe,hh,extra],-1)
        pred=self.candidate(feats.float())
        matched_similarity=-((qe-ce)**2).mean(-1).clamp(max=100)
        return {
            'logit_bias':pred[...,0]+0.02*matched_similarity,
            'normal_bias':pred[...,1],
            'failure_bias':pred[...,2],
            'expected_gain':pred[...,3],
            'expected_harm':torch.nn.functional.softplus(pred[...,4]),
            'query_embedding':q,
            'support':support[...,0],
            'state_entropy':entropy,
        }

    def mixture_gain(self,query_embedding,base,proposal,support,disagreement=None):
        b,p,n=base.shape
        hh=self.horizon_embed(torch.arange(p,device=base.device))[None,:,None].expand(b,p,n,-1)
        if disagreement is None:disagreement=torch.zeros_like(base)
        features=torch.cat([query_embedding[:,None].expand(b,p,n,-1),hh,
              torch.stack([base,proposal,support[:,None].expand_as(base),disagreement,proposal.abs()],-1)],-1)
        return self.mixture(features.float())
