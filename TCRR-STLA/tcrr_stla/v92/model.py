"""v9.2.0 end-to-end CF student; no current-query full data or labels in forward.

Normal branch remains exactly the v9.1 algorithm and is explicitly frozen
in every CF stage. Learned components operate only on a historical CF response.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..v9.model import TCRRV9
from .cf_encoder import TemporalCFEncoder, MultiRelationCFGraph
from . import VERSION

class TCRRV92(nn.Module):
    def __init__(self,graph,context_dim,hidden=32,cf_hidden=16,blocks=3,bound=3.,max_align=1.5,temp=.2):
        super().__init__()
        self.normal_core=TCRRV9(graph,context_dim,hidden=hidden,blocks=blocks,bound=bound)
        # Reuse the exact, already validated v9.1 Normal algorithm while
        # dropping every legacy CF parameter: a report must not list unused
        # v9.1 heads as trainable mechanisms in v9.2.
        for unused in ('temporal','cf_local','cf_fusion','amplitude','utility','fullinfo','none','temp'):
            delattr(self.normal_core,unused)
        self.blocks=int(blocks);self.bound=float(bound);self.max_align=float(max_align)
        self.temporal_cf=TemporalCFEncoder(blocks,cf_hidden)
        self.cf_graph=MultiRelationCFGraph(cf_hidden,hidden)
        self.cf_fusion=nn.Sequential(nn.Linear(hidden*2+cf_hidden*2+6,hidden),nn.GELU(),
                                     nn.Linear(hidden,hidden),nn.GELU())
        self.alignment=nn.Linear(hidden,1)
        self.utility=nn.Sequential(nn.Linear(hidden+3,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.null_head=nn.Sequential(nn.Linear(hidden+5,hidden),nn.GELU(),nn.Linear(hidden,blocks))
        self.full_info_head=nn.Linear(hidden,blocks)
        self.recoverability_scale=nn.Parameter(torch.tensor(0.))
        self.register_buffer('temperature',torch.tensor(float(temp)))
        nn.init.zeros_(self.alignment.weight)
        initial=.10/self.max_align
        nn.init.constant_(self.alignment.bias,math.atanh(initial))
        nn.init.zeros_(self.null_head[-1].weight)
        nn.init.constant_(self.null_head[-1].bias,0.)

    @property
    def graph(self):return self.normal_core.graph

    def bound_fn(self,x):return self.bound*torch.tanh(x/self.bound)

    def forward(self,*,base,context,x,mask,normal,cf,score,valid,match=None,
                enable_cf=True,alignment_mode='signed',rank_mode='learned',
                graph_mode='gnn_reliability',temporal_mode='cnn',
                null_mode='conditional',cf_strength=1.,recoverability_enabled=True):
        if base.ndim!=3 or normal.ndim!=4:raise ValueError('Expected base [B,H,N], normal [B,K,H,N]')
        b,h,n=base.shape;k=normal.shape[1];G=self.blocks
        if h%G or n!=self.graph.nodes or normal.shape!=(b,k,h,n) or cf.shape!=normal.shape:
            raise ValueError('v9.2 base/CF shape mismatch')
        if context.ndim!=3 or context.shape[:2]!=(b,n) or x.shape!=mask.shape or x.shape[0]!=b or x.shape[-1]!=n:
            raise ValueError('Query input shape mismatch')
        if score.shape!=(b,k) or valid.shape!=(b,k):raise ValueError('Candidate index/score shapes mismatch')
        if match is None:match=base.new_zeros((b,k,n,2))
        if match.shape!=(b,k,n,2):raise ValueError('Candidate matching shape mismatch')
        valid=valid.bool()
        if not (torch.isfinite(base).all() and torch.isfinite(normal).all() and
                torch.isfinite(cf).all() and torch.isfinite(score[valid]).all()):
            raise ValueError('Non-finite model evidence; check source cache, replay and candidate scores')
        if cf_strength<0:raise ValueError('cf_strength must be nonnegative')
        failure=self.normal_core.failure_features(mask)
        graph_q=self.graph.query_summary(x,mask).to(context.dtype)
        q=self.normal_core.query(torch.cat([failure,graph_q,context],-1))
        nraw,kz,wn=self.normal_core.normal_anchor(base,normal,q,match,score,valid)
        normal_pred=base+self.bound_fn(nraw)
        if not enable_cf:
            return {'prediction':normal_pred,'normal_prediction':normal_pred,
                    'base':base,'normal_raw':nraw,'cf_raw':torch.zeros_like(nraw),
                    'cf_increment':torch.zeros_like(nraw),'normal_weights':wn}
        local=self.temporal_cf(normal,cf,mode=temporal_mode)
        graph_cf,graph_gates=self.cf_graph(local,q,graph_q,self.graph,mode=graph_mode)
        rel=graph_q[...,[1,4,7]][:,None,None].expand(b,k,G,n,3)
        qg=q[:,None,None].expand(b,k,G,n,-1)
        kg=kz[:,:,None].expand(b,k,G,n,-1)
        z=self.cf_fusion(torch.cat([qg,kg,local,graph_cf,rel,graph_gates[:,None,None].expand(b,k,G,n,3)],-1))
        raw_alpha=self.alignment(z).squeeze(-1)
        if alignment_mode=='signed':alpha=self.max_align*torch.tanh(raw_alpha)
        elif alignment_mode=='positive':alpha=1.+.5*torch.tanh(raw_alpha)
        elif alignment_mode=='raw':alpha=torch.ones_like(raw_alpha)
        elif alignment_mode=='off':alpha=torch.zeros_like(raw_alpha)
        else:raise ValueError(f'Unknown alignment mode: {alignment_mode}')
        span=h//G
        candidate_cf=cf*alpha.repeat_interleave(span,2)
        block_cf=candidate_cf.reshape(b,k,G,span,n)
        u_features=torch.stack([block_cf.mean(3),block_cf.abs().mean(3),
                                block_cf.std(3,unbiased=False)],-1)
        two_util=self.utility(torch.cat([z,u_features],-1))
        single_util=two_util[...,0].float();marginal_util=two_util[...,1].float()
        recoverability=self.full_info_head(q).permute(0,2,1).float()
        if null_mode=='conditional':
            null=self.null_head(torch.cat([q,failure],-1)).permute(0,2,1).float()
            if recoverability_enabled:
                null=null-torch.tanh(self.recoverability_scale)*recoverability
        elif null_mode=='fixed':null=single_util.new_zeros((b,G,n))
        else:raise ValueError(f'Unknown null mode {null_mode}')
        if rank_mode=='learned':logits=single_util/self.temperature.float().clamp_min(.025)+score[:,:,None,None].float()
        elif rank_mode=='uniform':logits=torch.zeros_like(single_util)
        elif rank_mode=='similarity':logits=score[:,:,None,None].float().expand_as(single_util)
        else:raise ValueError(f'Unknown rank mode {rank_mode}')
        logits=logits.masked_fill(~valid[:,:,None,None],-1e4)
        weights=torch.softmax(torch.cat([logits,null[:,None]],1).float(),dim=1)
        wc=weights[:,:k].repeat_interleave(span,2).to(candidate_cf.dtype)
        cfraw=(wc*candidate_cf).sum(1)*float(cf_strength)
        pred=base+self.bound_fn(nraw+cfraw)
        return {'base':base,'normal_prediction':normal_pred,'prediction':pred,
                'normal_raw':nraw,'cf_raw':cfraw,'cf_increment':pred-normal_pred,
                'candidate_cf':candidate_cf,'cf_utility':single_util,
                'marginal_utility':marginal_util,'cf_weights':weights,
                'normal_weights':wn,'alignment':alpha,'null_logit':null,
                'graph_gates':graph_gates,'cf_local_features':local,
                'cf_graph_features':graph_cf,'recoverability':recoverability,
                'bound':self.bound}
