"""PCFT v7 unified research model. No evaluation targets enter forward()."""
from __future__ import annotations
import torch
from torch import nn
from .backbone import STLAAnchor
from .support import ForecastSupport
from .propagation import SparsePropagation
from .utility import UtilityController


def head(d,extra=0):
    # Zero output after construction: initial full prediction = exact official STLA.
    h=nn.Sequential(nn.Linear(d*2+extra,d),nn.GELU(),nn.Linear(d,1))
    nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
    return h

class PCFTV7(nn.Module):
    STAGES=('local','propagation','utility','joint')
    def __init__(self,base,graph,target_idx,channels=3,hidden=64,horizon=12,controller_dim=32,enable_functional=False):
        super().__init__()
        self.anchor=STLAAnchor(base)
        dim=int(base.model_dim)
        self.support=ForecastSupport(dim,channels,horizon,hidden)
        self.propagation=SparsePropagation(dim,graph,target_idx,channels,functional=enable_functional)
        self.local_head=head(dim,extra=2)
        self.prop_head=head(dim,extra=2)
        self.controller=UtilityController(dim,controller_dim)
        self.target_idx=int(target_idx)
        self.stage='joint'
        self.ablation={'use_support':True,'use_propagation':True,'use_controller':True,
                       'use_direction':True,'use_local':True,'use_null_source':True}
    def set_stage(self,stage):
        if stage not in self.STAGES:raise ValueError(stage)
        self.stage=stage
    def set_ablation(self,**opts):
        for k,v in opts.items():
            if k not in self.ablation:raise ValueError(k)
            self.ablation[k]=bool(v)
    def forward(self,batch,*,diagnostic=False,edge_keep_mask=None,coverage_intervention=False,trace_edges=False):
        anchor=self.anchor(batch)
        y0=anchor['prediction'].float()
        h=anchor['node_features']
        B,P,N=y0.shape
        if not self.ablation['use_support']:
            support={'node_context':h,'forecast_context':h[:,None].expand(-1,P,-1,-1),
                'source_quality':torch.ones(B,N,1,device=h.device),
                'support_score':torch.ones(B,P,N,1,device=h.device)}
        else:support=self.support(batch,h,y0,self.target_idx)
        ctx=support['forecast_context']
        local=(self.local_head(torch.cat((ctx,support['node_context'][:,None].expand_as(ctx),y0[...,None],support['support_score']),-1)).squeeze(-1)
               if self.ablation['use_local'] else torch.zeros_like(y0))
        # A warm-up stage avoids source and policy overhead.
        do_prop=(self.stage!='local' and self.ablation['use_propagation'])
        if do_prop:
            p=self.propagation(batch,support,use_direction=self.ablation['use_direction'],
                               use_null_source=self.ablation['use_null_source'],
                               return_diagnostics=diagnostic,edge_keep_mask=edge_keep_mask,trace_edges=trace_edges)
            coverage=p['coverage']
            prop=self.prop_head(torch.cat((ctx,p['message'],y0[...,None],coverage),-1)).squeeze(-1)
            if not coverage_intervention:prop=prop*coverage[...,0]
            # A fully unavailable source may never introduce a learned bias-only message.
            prop=prop*(coverage[...,0]>0).float()
        else:
            coverage=y0.new_zeros(B,P,N,1);prop=torch.zeros_like(local)
        candidates=torch.stack((y0,y0+local,y0+local+prop),-1)
        avail=batch['observed_mask'][...,self.target_idx].mean(1).unsqueeze(-1).float()
        if self.stage in ('local','propagation') or not self.ablation['use_controller']:
            index=1 if self.stage=='local' else 2
            weights=torch.zeros_like(candidates);weights[...,index]=1.
            policy={'weights':weights,'predicted_gain':y0.new_zeros(B,P,N,2),
                    'predicted_harm_logits':y0.new_zeros(B,P,N,2)}
        else:
            policy=self.controller(ctx,y0,local,prop,coverage,avail,support['support_score'])
        final=(policy['weights']*candidates).sum(-1)
        result={'prediction':final,'baseline':y0,'candidates':candidates,
                'local_delta':local,'prop_delta':prop,'policy':policy,
                'coverage':coverage,'support':support}
        if diagnostic:
            result['propagation_trace']=p.get('trace') if do_prop else None
        return result
