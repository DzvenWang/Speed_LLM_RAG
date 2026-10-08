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
        self.controller=UtilityRiskController(cfg['model']['hidden'],self.p)
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
    def forward(self,inputs,memory,mode='full'):
        x=inputs['x_obs'].float();mask=inputs['obs_mask'].bool();tod=inputs['tod'];dow=inputs['dow']
        base=self.stla(x,mask,tod,dow,enabled=False,return_hidden=True)
        if mode=='retrieval_off':
            return {'prediction':base['prediction'],'base_prediction':base['prediction'],'candidate_prediction':base['prediction']}
        ctx=self.regions(x,mask);obs=self.observability(ctx)
        retrieved=self.retrieval(ctx,self.regions,memory,inputs['window_start'],mode=mode)
        state=self.state_recovery(inputs,ctx,retrieved,self.regions,memory)
        if mode=='innovation_only':
            state=dict(state);state['gate']=torch.zeros_like(state['gate']);state['supported']=x
        latent=self.innovation(inputs,state,retrieved,self.regions,memory,base['hidden'])
        if mode=='state_only':latent['delta']=torch.zeros_like(latent['delta'])
        conf=self.reliability(inputs,ctx,retrieved,state,latent,self.regions)
        supported=self.stla(x,mask,tod,dow,state=state,innovation=latent['delta'],confidence=conf,retrieved_tokens=latent['support_tokens'],support_valid=latent['support_valid'],enabled=True,return_hidden=True)
        # Forecast residual is derived from retrieved innovation, not a second predictor.
        compact=torch.cat([latent['delta'].mean(1),state['uncertainty'].mean(1)[...,None],conf.mean(1)[...,None]],-1)
        residual=self.forecast_residual(compact).transpose(1,2)
        valid=retrieved['group_valid'][:,self.regions.owner][:,None,:]
        candidate=supported['prediction']+residual*valid
        ctr=self.controller(base['prediction'],candidate,conf,latent,state,retrieved,self.regions)
        final_prediction=ctr['prediction']
        if mode=='no_risk':
            simple_gate=conf.mean(1)[:,None,:].expand_as(candidate).clamp(0,1)*retrieved['group_valid'][:,self.regions.owner][:,None,:]
            final_prediction=base['prediction']+simple_gate*(candidate-base['prediction'])
            ctr['gate']=simple_gate
        if hasattr(self,'calibration_multiplier'):
            final_prediction=base['prediction']+self.calibration_multiplier*ctr['gate']*(candidate-base['prediction'])
        return {'prediction':final_prediction,'base_prediction':base['prediction'],'candidate_prediction':candidate,'recovered_state':state['recovered'],'state_uncertainty':state['uncertainty'],'state_gate':state['gate'],'supported_history':state['supported'],'latent_innovation':latent['delta'],'innovation_uncertainty':latent['uncertainty'],'retrieved_ids':retrieved['ids'],'retrieved_weights':retrieved['weights'],'retrieval_valid':retrieved['group_valid'],'retrieval_confidence':conf,'predicted_utility':ctr['utility'],'predicted_risk':ctr['risk'],'control_raw_features':ctr['raw_features'],'final_gate':ctr['gate'],'retrieval_query':retrieved['query']}
