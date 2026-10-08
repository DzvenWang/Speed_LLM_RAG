from __future__ import annotations
import copy
import torch
from torch import nn
from .graph import GraphRegions,ObservabilityEncoder
from .retrieval import TargetRetriever
from .recovery import RetrievedStateRecovery,InnovationStudent
from .control import Reliability,UtilityRiskController
from .backbone import FARESTLA

class FARE(nn.Module):
    def __init__(self,stla_base,adj,cfg,directed=None):
        super().__init__();self.cfg=cfg;self.n=int(stla_base.num_nodes);self.h=int(stla_base.in_steps);self.p=int(stla_base.out_steps)
        gc=cfg['graph'];r=cfg['retrieval'];inc=cfg['innovation']
        self.regions=GraphRegions(adj,gc['target_region_size'],gc['max_anchors'],directed=directed)
        self.stla=FARESTLA(stla_base,self.n,adj,cfg['backbone'])
        self.observability=ObservabilityEncoder(cfg['model']['hidden'])
        self.retrieval=TargetRetriever(self.h,cfg['model']['hidden'],r['key_dim'],r['coarse_topk'],r['topk'],overlap=self.h+self.p)
        self.state_recovery=RetrievedStateRecovery(cfg['model']['hidden'])
        self.innovation=InnovationStudent(self.h,self.stla.d,inc['memory_dim'],inc['rank'],inc['heads'])
        self.reliability=Reliability(cfg['model']['hidden'])
        self.controller=UtilityRiskController(cfg['model']['hidden'],self.p,**cfg.get('control',{}))
        self.forecast_residual=nn.Sequential(nn.Linear(self.stla.d+2,self.stla.d),nn.GELU(),nn.Linear(self.stla.d,self.p))
        nn.init.zeros_(self.forecast_residual[-1].weight);nn.init.zeros_(self.forecast_residual[-1].bias)
        # fixed pretrained snapshot for counterfactual representation targets
        self.teacher=copy.deepcopy(stla_base)
        for p in self.teacher.parameters():p.requires_grad=False
        self.teacher.eval()
        self.register_buffer('calibration_multiplier',torch.tensor(1.))
        self.set_phase('retrieval')
    def train(self,mode=True):
        super().train(mode)
        self.teacher.eval()
        if not any(p.requires_grad for p in self.stla.base.parameters()):self.stla.base.eval()
        return self
    def set_phase(self,phase):
        self.phase=phase;self.stla.set_phase(phase)
        for name,module in [('observability',self.observability),('retrieval',self.retrieval),('state_recovery',self.state_recovery),('innovation',self.innovation),('reliability',self.reliability),('controller',self.controller),('forecast_residual',self.forecast_residual)]:
            for p in module.parameters():p.requires_grad=True
        if phase=='retrieval':
            for name in ('state_recovery','innovation','reliability','controller','forecast_residual'):
                for p in getattr(self,name).parameters():p.requires_grad=False
        if phase=='recovery':
            for name in ('controller','forecast_residual'):
                for p in getattr(self,name).parameters():p.requires_grad=False
        if self.cfg['experiment'].get('training_mode','full')=='no_memory':
            # Matched no-memory architecture: only locally observed STLA
            # embeddings, decomposition and graph adapters can be trained.
            for name in ('observability','retrieval','state_recovery','innovation',
                         'reliability','controller','forecast_residual'):
                for p in getattr(self,name).parameters():p.requires_grad=False
            for name in ('retrieval_cross','retrieval_cross_out','cross_adapter'):
                for p in getattr(self.stla,name).parameters():p.requires_grad=False
        if phase=='calibration':
            for name in ('retrieval','observability','state_recovery','innovation'):
                for p in getattr(self,name).parameters():p.requires_grad=False
    @torch.no_grad()
    def refresh(self,memory):
        self.retrieval.refresh(memory,self.regions)
    def teacher_hidden(self,x,mask,tod,dow):
        # This is only called by the loss module; no supervision passed to inference forward.
        base=self.stla.base;self.stla.base=self.teacher
        try:
            with torch.no_grad():return self.stla(x,mask,tod,dow,enabled=False)['hidden']
        finally:self.stla.base=base
    def forward(self,inputs,memory,mode='full',retrieval_override=None):
        """Inference-safe forward. Only observed history, mask, time, start enter.

        Strict fixed-checkpoint interventions:
          no_memory: same enabled FARE-STLA input/adapters, no memory calls
          state_only: state support, but NO innovation student, tokens or residual
          innovation_only: no retrieved state used anywhere in innovation inputs
          no_risk/no_utility: identical bounded controller minus the named signal
        These interventions DO NOT replace matched retrained ablations.
        """
        x=inputs['x_obs'].float();mask=inputs['obs_mask'].bool()
        tod=inputs['tod'];dow=inputs['dow']
        base=self.stla(x,mask,tod,dow,enabled=False,return_hidden=True)
        if mode=='retrieval_off':
            return {'prediction':base['prediction'],
                    'base_prediction':base['prediction'],
                    'candidate_prediction':base['prediction']}
        if mode=='no_memory':
            # Keep the SAME FARE-STLA feature and adapter architecture, enable
            # mask/graph/decomposition features, but provide no external evidence.
            masked=self.stla(x,mask,tod,dow,enabled=True,return_hidden=True)
            y=masked['prediction']
            return {'prediction':y,'base_prediction':base['prediction'],
                    'candidate_prediction':y,'matched_no_memory_prediction':y,
                    'final_gate':torch.zeros_like(y)}
        ctx=self.regions(x,mask)
        observability=self.observability(ctx)
        retrieved=(self.retrieval(ctx,self.regions,memory,inputs['window_start'],mode=mode,observability=observability)
                   if retrieval_override is None else retrieval_override)
        if retrieval_override is not None:
            # Only precomputed inference-safe legal candidate pools are admissible.
            legal=(inputs['window_start'][:,None,None]-memory.starts[retrieved['ids']]).abs()>=self.h+self.p
            if not torch.all(~retrieved['valid'] | legal):
                raise ValueError('Fixed-pool retrieval override contains overlapping history/forecast')
            if not torch.all(~retrieved['valid'] | (retrieved['weights']>=0)):
                raise ValueError('Invalid fixed-pool retrieval weights')
        # No training labels reach any retrieval feature.
        strict_innovation_only=(mode=='innovation_only')
        if strict_innovation_only:
            # Do NOT even execute State Recovery in this independent intervention.
            # Student queries come from observed x and legal retrieved latent only.
            state={'recovered':x,'uncertainty':torch.zeros_like(x),
                   'gate':torch.zeros_like(x),'supported':x}
        else:
            state=self.state_recovery(inputs,ctx,retrieved,self.regions,memory,observability=observability)
        strict_state_only=(mode=='state_only')
        if strict_state_only:
            # Skip the student entirely; the old 'state_only' accidentally left
            # innovation-derived support tokens in the STLA Cross-Attention.
            latent={'delta':base['hidden'].new_zeros(base['hidden'].shape),
                    'uncertainty':torch.zeros_like(x),
                    'support_tokens':None,'support_valid':None}
        else:
            latent=self.innovation(inputs,state,retrieved,self.regions,memory,base['hidden'])
        conf=self.reliability(inputs,ctx,retrieved,state,latent,self.regions,observability=observability)
        enabled_state=None if strict_innovation_only else state
        supported=self.stla(x,mask,tod,dow,state=enabled_state,
                            innovation=None if strict_state_only else latent['delta'],
                            confidence=conf,
                            retrieved_tokens=latent['support_tokens'],
                            support_valid=latent['support_valid'],
                            enabled=True,return_hidden=True)
        if strict_state_only:
            residual=torch.zeros_like(base['prediction'])
        else:
            compact=torch.cat([latent['delta'].mean(1),
                               state['uncertainty'].mean(1)[...,None],
                               conf.mean(1)[...,None]],-1)
            residual=self.forecast_residual(compact).transpose(1,2)
        valid=retrieved['group_valid'][:,self.regions.owner][:,None,:]
        candidate=supported['prediction']+residual*valid
        control_opts={}
        if mode=='no_risk':control_opts['ignore_risk']=True
        if mode=='no_utility':control_opts['ignore_utility']=True
        if mode=='constant_alpha':
            control_opts['constant_alpha']=self.cfg.get('ablation',{}).get('constant_alpha',.12)
        ctr=self.controller(base['prediction'],candidate,conf,latent,state,
                            retrieved,self.regions,**control_opts)
        effective_gate=(self.calibration_multiplier*ctr['gate']).clamp(0,self.controller.max_alpha)
        final_prediction=base['prediction']+effective_gate*(candidate-base['prediction'])
        delta_norm=latent['delta'].abs().mean((-1,-2,-3))
        tokens=latent.get('support_tokens')
        token_norm=(tokens.abs().mean((-1,-2,-3)) if tokens is not None
                    else torch.zeros_like(delta_norm))
        return {'prediction':final_prediction,'base_prediction':base['prediction'],
                'candidate_prediction':candidate,
                'recovered_state':state['recovered'],
                'state_uncertainty':state['uncertainty'],
                'state_gate':state['gate'],
                'supported_history':state['supported'],
                'latent_innovation':latent['delta'],
                'innovation_uncertainty':latent['uncertainty'],
                'retrieved_ids':retrieved['ids'],
                'retrieved_weights':retrieved['weights'],
                'retrieved_valid_candidates':retrieved['valid'],
                'retrieval_valid':retrieved['group_valid'],
                'retrieval_confidence':conf,
                'predicted_utility':ctr['utility'],
                'predicted_risk':ctr['risk'],
                'control_raw_features':ctr['raw_features'],
                'final_gate':effective_gate,
                'correction_delta':candidate-base['prediction'],
                'retrieval_query':retrieved['query'],
                'observability_embedding':observability,
                'innovation_delta_abs':delta_norm,
                'retrieved_token_abs':token_norm}
