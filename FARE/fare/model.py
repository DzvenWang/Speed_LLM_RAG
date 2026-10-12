"""FARE v2.1: retrieval-driven forecasting, not memory-independent correction.

Inference never accesses the current target.  Historical outcomes are admissible
only after the memory episode ended before query issue time.
"""
from __future__ import annotations
import torch
from torch import nn
from .graph import GraphRegions
from .backbone import FARESTLA
from .retrieval import HierarchicalForecastRetriever


class FARE(nn.Module):
    def __init__(self,stla_base,adj,cfg,directed=None):
        super().__init__()
        self.cfg=cfg
        self.n=int(stla_base.num_nodes);self.h=int(stla_base.in_steps);self.p=int(stla_base.out_steps)
        self.regions=GraphRegions(adj,cfg['graph']['target_region_size'],cfg['graph']['max_anchors'],directed=directed)
        self.stla=FARESTLA(stla_base,self.n,adj,cfg['backbone'])
        r=cfg['retrieval'];dim=cfg['model']['hidden'];latent=cfg['innovation']['memory_dim']
        self.retrieval=HierarchicalForecastRetriever(
            self.h,self.p,self.stla.d,dim,r['key_dim'],r['coarse_topk'],
            r['shortlist_topk'],r['topk'],latent,self.h+self.p)
        self.state_projection=nn.Linear(dim,self.stla.d,bias=False)
        self.forecast_projection=nn.Linear(dim,self.stla.d,bias=False)
        # Retrieval tokens are the ONLY path to these extra predictions.
        self.final_control=nn.Sequential(nn.Linear(6,dim),nn.GELU(),nn.Linear(dim,self.p))
        nn.init.zeros_(self.final_control[-1].weight)
        nn.init.constant_(self.final_control[-1].bias,-0.3)
        self.local_control=nn.Sequential(nn.Linear(self.stla.d+2,dim),nn.GELU(),nn.Linear(dim,self.p))
        nn.init.zeros_(self.local_control[-1].weight)
        nn.init.zeros_(self.local_control[-1].bias)
        self.register_buffer('calibration_multiplier',torch.tensor(1.))
        self.phase='evidence'
        for p in self.stla.base.parameters():p.requires_grad=False
        self.set_phase('evidence')

    def train(self,mode=True):
        super().train(mode)
        self.stla.base.eval()
        return self

    @torch.no_grad()
    def teacher_hidden(self,x,mask,tod,dow):
        return self.stla(x,mask,tod,dow,enabled=False,return_hidden=True)['hidden']

    @torch.no_grad()
    def refresh(self,memory):
        self.retrieval.refresh(memory,self.regions)

    def set_phase(self,phase):
        allowed=('evidence','retrieval','integration','control')
        if phase not in allowed:raise ValueError(f'Unknown v2.1 training phase {phase}')
        self.phase=phase
        for p in self.parameters():p.requires_grad=False
        active={
            'evidence':('retrieval.adapter','retrieval.probe'),
            'retrieval':('retrieval.encoder','retrieval.fast','retrieval.utility',
                         'retrieval.memory_latent_projection','retrieval.log_temperature','retrieval.null_logit'),
            'integration':('retrieval.encoder','retrieval.fast','retrieval.utility',
                          'retrieval.adapter','retrieval.memory_latent_projection',
                          'retrieval.log_temperature','retrieval.null_logit',
                          'state_projection','forecast_projection','stla.retrieval_cross',
                          'stla.retrieval_cross_out','final_control'),
            'control':('final_control',)
        }[phase]
        for name,p in self.named_parameters():
            if name.startswith(active):p.requires_grad=True
        for p in self.stla.base.parameters():p.requires_grad=False
        if self.cfg['experiment'].get('training_mode')=='no_memory':
            for p in self.parameters():p.requires_grad=False
            for p in self.local_control.parameters():p.requires_grad=True
        return self

    def _base(self,inputs):
        return self.stla(inputs['x_obs'].float(),inputs['obs_mask'].bool(),
            inputs['tod'],inputs['dow'],enabled=False,return_hidden=True)

    def forward(self,inputs,memory,mode='full',return_diagnostics=False,phase_fast=None):
        base=self._base(inputs)
        y0=base['prediction']
        if mode=='retrieval_off':
            return {'prediction':y0,'base_prediction':y0,'candidate_prediction':y0,
                    'final_gate':torch.zeros_like(y0)}
        if mode=='no_memory' or self.cfg['experiment'].get('training_mode')=='no_memory':
            cov=inputs['obs_mask'].float().mean(1)
            x=self.local_control(torch.cat([base['hidden'].mean(1),cov[...,None],
                y0.std(1,unbiased=False)[...,None]],-1)).transpose(1,2)
            cand=y0+x
            return {'prediction':cand,'base_prediction':y0,'candidate_prediction':cand,
                    'final_gate':torch.zeros_like(y0)}
        ctx=self.regions(inputs['x_obs'].float(),inputs['obs_mask'].bool())
        retrieved=self.retrieval(inputs,base,ctx,self.regions,memory,mode=mode)
        weights=retrieved['weights']
        ev=retrieved['selected_evidence']
        # Node-specific historical state evidence is auxiliary, not obligatory.
        state=(ev['state_history']*weights[...,None]).sum(2)
        x=inputs['x_obs'].permute(0,2,1)
        observed=inputs['obs_mask'].permute(0,2,1)
        recovered=torch.where(observed,x,state)
        token=self.forecast_projection(retrieved['tokens'])+\
              self.state_projection(ev['state_token']*weights[...,None])
        valid=retrieved['valid'] & (weights>0)
        supported=self.stla(inputs['x_obs'].float(),inputs['obs_mask'].bool(),
              inputs['tod'],inputs['dow'],enabled=True,
              retrieved_tokens=token,support_valid=valid,return_hidden=True)
        injected=supported['prediction']-y0
        forecast_delta=retrieved['forecast_delta'].transpose(1,2)
        candidate=y0+injected+forecast_delta
        reliability=(ev['reliability']*weights).sum(-1)
        utility=(retrieved['utility']*weights).sum(-1)
        risk=(retrieved['risk']*weights).sum(-1)
        present=weights.sum(-1)
        features=torch.stack([present,reliability,utility,risk,
               forecast_delta.abs().mean(1),
               inputs['obs_mask'].float().mean(1)],-1)
        gate=torch.sigmoid(self.final_control(features)).transpose(1,2)
        gate=gate*present[:,None,:]*self.calibration_multiplier
        final=y0+gate*(candidate-y0)
        delta=(candidate-y0).abs().mean()
        return {'prediction':final,'base_prediction':y0,'candidate_prediction':candidate,
            'final_gate':gate,'retrieved_ids':retrieved['ids'],
            'retrieved_weights':weights,'retrieved_valid_candidates':retrieved['valid'],
            'retrieval_valid':retrieved['node_valid'],
            'retrieval_query':retrieved['region_query'],
            'retrieval_evidence':retrieved,
            'retrieval_null_probability':retrieved['null_weight'],
            'retrieval_utility_scores':retrieved['utility'],
            'retrieval_confidence':reliability,
            'predicted_utility':utility,'predicted_risk':risk,
            'recovered_state':recovered.transpose(1,2),
            'state_gate':(1-inputs['obs_mask'].float())*present[:,None,:],
            'state_uncertainty':(1-reliability)[:,None,:].expand_as(inputs['x_obs']),
            'latent_innovation':token.sum(2)[:,None,:,:].expand(-1,self.h,-1,-1),
            'innovation_uncertainty':(1-reliability)[:,None,:].expand_as(inputs['x_obs']),
            'innovation_delta_abs':token.abs().mean((1,2,3)),
            'retrieved_token_abs':token.abs().mean((1,2,3)),
            'correction_delta':candidate-y0,
            'probe_predictions':retrieved['probe_predictions'],
            'node_query':retrieved['node_query'],
            'shortlist_valid':retrieved['shortlist_valid'],
            'shortlist_scores':retrieved['shortlist_scores']}
