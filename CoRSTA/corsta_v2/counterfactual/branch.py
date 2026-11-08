"""Horizon-aware CF representation and internal STLA latent shift."""
from __future__ import annotations
import torch
from torch import nn


class CrossTime(nn.Module):
    """Node-local cross-attention. Query length can be L or P; no NxN dense attention."""
    def __init__(self,qdim,kvdim,out_dim,heads=4):
        super().__init__()
        self.q=nn.Linear(qdim,out_dim)
        self.kv=nn.Linear(kvdim,out_dim)
        self.attn=nn.MultiheadAttention(out_dim,heads,dropout=0.0,batch_first=True)
        self.out=nn.Sequential(nn.LayerNorm(out_dim),nn.Linear(out_dim,out_dim))

    def forward(self,query,key_value):
        B,Q,N,Dq=query.shape;Bk,K,Nk,Dk=key_value.shape
        if B!=Bk or N!=Nk:raise ValueError('CrossTime batch/node mismatch')
        q=self.q(query).permute(0,2,1,3).reshape(B*N,Q,-1)
        kv=self.kv(key_value).permute(0,2,1,3).reshape(B*N,K,-1)
        att,_=self.attn(q,kv,kv,need_weights=False)
        out=self.out(att).view(B,N,Q,-1).permute(0,2,1,3)
        return out


class CFBranch(nn.Module):
    def __init__(self,hidden=64,failure=16,dim=32,horizon=12,heads=4,shift_scale=.35):
        super().__init__()
        self.horizon=horizon;self.shift_scale=float(shift_scale)
        self.pos=nn.Embedding(horizon,dim)
        self.node_context=nn.Linear(failure,dim)
        self.p0_embed=nn.Linear(1,dim)
        self.forecast_query=nn.LayerNorm(dim)
        self.history_cross=CrossTime(dim,hidden+failure,dim,heads)
        self.graph_fusion=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,dim))
        self.reverse_cross=CrossTime(hidden,dim,hidden,heads)
        self.shift=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        nn.init.normal_(self.shift[-1].weight,mean=0,std=1e-3);nn.init.zeros_(self.shift[-1].bias)
        self.gate_head=nn.Linear(hidden,1)
        nn.init.zeros_(self.gate_head.weight);nn.init.constant_(self.gate_head.bias,1.5)
        self.scale_head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
        # Mask-effect head: expected full-view absolute-error advantage, predicted
        # ONLY from currently visible masked-input tokens, never from TRAIN labels.
        self.error_gain_head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),
                                           nn.GELU(),nn.Linear(dim,1))
        # Action head: whether this Student's *actual candidate* is beneficial.
        # Separate from teacher reference gain: P_full may itself be inaccurate.
        self.acceptance_head=nn.Sequential(nn.Linear(dim+3,dim),nn.GELU(),nn.Linear(dim,1))
        nn.init.zeros_(self.acceptance_head[-1].weight)
        nn.init.constant_(self.acceptance_head[-1].bias,2.7)

        # True gain of this candidate over a FROZEN normal residual anchor.
        # Distinct from the full-vs-masked teacher gain and v021 P0 acceptance.
        self.anchor_gain_head=nn.Sequential(nn.Linear(dim+3,dim),nn.GELU(),nn.Linear(dim,1))
        self.anchor_acceptance_head=nn.Sequential(nn.Linear(dim+4,dim),nn.GELU(),nn.Linear(dim,1))
        nn.init.zeros_(self.anchor_acceptance_head[-1].weight)
        nn.init.constant_(self.anchor_acceptance_head[-1].bias,6.0)

    def anchor_action(self,cf,p_normal,p_candidate):
        delta=p_candidate-p_normal
        features=torch.cat((cf['tokens'],cf['teacher_gain'][...,None],
                            delta[...,None],p_normal[...,None]),dim=-1)
        pred_gain=self.anchor_gain_head(features).squeeze(-1)
        accept_features=torch.cat((features,pred_gain[...,None]),dim=-1)
        logit=self.anchor_acceptance_head(accept_features).squeeze(-1)
        return pred_gain,logit,torch.sigmoid(logit)

    def temporal_tokens(self,h0,p0,failure):
        B,L,N,D=h0.shape;P=p0.shape[1]
        if P!=self.horizon:raise ValueError('CF horizon mismatch')
        f=failure['tokens']
        static=f.mean(1)
        steps=self.pos(torch.arange(P,device=h0.device))[None,:,None,:]
        q=self.forecast_query(steps+self.node_context(static)[:,None]+self.p0_embed(p0[...,None]))
        return self.history_cross(q,torch.cat([h0,f],dim=-1))

    def forward(self,h0,p0,failure,spatial,temporal=None):
        if temporal is None:temporal=self.temporal_tokens(h0,p0,failure)
        merged=self.graph_fusion(torch.cat([temporal,spatial['tokens']],dim=-1))
        raw=self.reverse_cross(h0,merged)
        shift=self.shift(raw)*self.shift_scale
        # Scale is learned as an auxiliary uncertainty parameter, not an arbitrary gate.
        uncertainty=torch.nn.functional.softplus(self.scale_head(merged).squeeze(-1))+1e-4
        teacher_gain=self.error_gain_head(merged).squeeze(-1)
        return {'tokens':merged,'temporal_tokens':temporal,'delta_h':shift,
                'latent_gate':torch.sigmoid(self.gate_head(raw)),
                'uncertainty':uncertainty,'spatial':spatial,
                'teacher_gain':teacher_gain}

    def acceptance(self,cf,p0,p_candidate):
        """Inference-only features; predicts whether THIS candidate is worth applying."""
        delta=p_candidate-p0
        features=torch.cat((cf['tokens'],cf['teacher_gain'][...,None],
                            delta[...,None],p0[...,None]),dim=-1)
        logits=self.acceptance_head(features).squeeze(-1)
        return logits,torch.sigmoid(logits)
