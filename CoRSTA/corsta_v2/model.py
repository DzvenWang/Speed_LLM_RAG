"""CoRSTA v0.2: exactly CF and Forecast Residual are prediction mechanisms.
No retrieval, no query futures, no full-view history anywhere in forward().

v0.2.3 (normal_anchored only) adds two OPTIONAL deployment mechanisms that sit
on top of the CF shift and are off unless configured:
  B  Counterfactual Dose Ladder: monotone dose-response psi_d=d**gamma over the
     CF shift plus a dose policy picking, per forecast element, the dose with the
     largest predicted real gain (or no correction).
  A  View-consistency reliability: extra views drop further observed entries;
     cross-view agreement of corrected vs uncorrected forecasts is a label-free
     reliability feature for the dose policy.
"""
from __future__ import annotations
import torch
from torch import nn
from corsta_v2.data.features import FailureEncoder
from corsta_v2.counterfactual.branch import CFBranch
from corsta_v2.residual.branch import ResidualBranch
from corsta_v2.spatial.graph import GraphEvidence
from corsta_v2.control.controller import ResidualBenefitController
from corsta_v2.stla.backbone import base_features,decode
from corsta_v2.counterfactual.dose import (CONSISTENCY_DIM,validate_levels,dose_psi,
                                           consistency_features,select_dose)


class CoRSTAv2(nn.Module):
    def __init__(self,stla,graph,hidden=64,failure_dim=16,cf_dim=32,residual_dim=32,
                 horizon=12,heads=4,latent_scale=.35,cf_error_gate=True,prediction_mode='cf_first',
                 dose_levels=(),dose_margin=0.0,dose_temperature=0.01,
                 consistency_views=0,consistency_drop=0.2,consistency_seed=10):
        super().__init__()
        dose_levels=validate_levels(dose_levels) if len(tuple(dose_levels)) else ()
        if (dose_levels or int(consistency_views)>0) and prediction_mode!='normal_anchored':
            raise ValueError('Dose ladder / view consistency require prediction_mode=normal_anchored')
        if int(consistency_views)<0 or not 0.0<=float(consistency_drop)<1.0:
            raise ValueError('consistency_views>=0 and 0<=consistency_drop<1 required')
        if float(dose_temperature)<=0:raise ValueError('dose_temperature must be positive')
        self.consistency_views=int(consistency_views)
        self.consistency_drop=float(consistency_drop)
        self.consistency_seed=int(consistency_seed)
        self.dose_margin=float(dose_margin)
        self.dose_temperature=float(dose_temperature)
        self.dose_soft_training=True
        self.stla=stla
        self.stla.requires_grad_(False)
        self.stla.eval()
        self.failure=FailureEncoder(failure_dim)
        self.input_adapter=nn.Linear(failure_dim,hidden)
        nn.init.normal_(self.input_adapter.weight,std=1e-3)
        nn.init.zeros_(self.input_adapter.bias)
        self.cf=CFBranch(hidden,failure_dim,cf_dim,horizon,heads,latent_scale,
                         dose_levels=dose_levels,
                         consistency_dim=(CONSISTENCY_DIM if self.consistency_views>0 else 0))
        self.dose_enabled=bool(dose_levels)
        self.residual=ResidualBranch(hidden,failure_dim,cf_dim,residual_dim,horizon,heads,latent_scale)
        self.graph_cf=GraphEvidence(graph,dim=cf_dim)
        self.graph_residual=GraphEvidence(graph,dim=residual_dim)
        self.controller=ResidualBenefitController(cf_dim,residual_dim)
        if prediction_mode not in ('cf_first','normal_anchored'):
            raise ValueError('Unsupported prediction mode')
        self.prediction_mode=prediction_mode
        self.stage='cf' if prediction_mode=='cf_first' else 'normal'
        self.use_graph=False
        self.cf_error_gate=bool(cf_error_gate)

    def train(self,mode=True):
        super().train(mode)
        self.stla.eval() # Keep frozen pretrained STLA dropout disabled.
        if self.prediction_mode=='normal_anchored' and self.stage in ('anchor_cf','anchor_control'):
            # Frozen normal anchor must retain its exact inference behaviour.
            self.failure.eval();self.residual.eval();self.graph_residual.eval()
        if self.prediction_mode=='normal_anchored' and self.stage=='anchor_control':
            self.cf.eval();self.graph_cf.eval();self.input_adapter.eval()
        return self

    def set_stage(self,stage,use_graph=None):
        accepted=(('cf','residual','joint','control') if self.prediction_mode=='cf_first'
                  else ('normal','anchor_cf','anchor_control'))
        if stage not in accepted:raise ValueError('Unknown training stage')
        self.stage=stage
        if use_graph is not None:self.use_graph=bool(use_graph)
        trainable={
            'normal':('failure','residual','graph_residual'),
            'anchor_cf':('cf','input_adapter','graph_cf'),
            'anchor_control':(),
            'cf':('failure','input_adapter','cf'),
            'residual':('residual',),
            'joint':('failure','input_adapter','cf','residual','graph_cf','graph_residual'),
            'control':('controller',),
        }[stage]
        for name,module in self.named_children():
            for p in module.parameters():p.requires_grad_(name in trainable)
        self.stla.requires_grad_(False)
        if stage=='anchor_control':
            # Only gain regression and the policy are optimized; the CF
            # candidate and frozen normal anchor are immutable.
            self.cf.anchor_gain_head.requires_grad_(True)
            self.cf.anchor_acceptance_head.requires_grad_(True)
            if self.dose_enabled:
                # The dose policy is a deployment head: trained here, on top of
                # the immutable CF candidate and Normal anchor.
                self.cf.dose_policy.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    def _no_graph(self,token):
        B,P,N,D=token.shape
        return {'tokens':torch.zeros_like(token),
                'confidence':token.new_zeros((B,P,N)),
                'null_weight':token.new_ones((B,P,N))}

    def forward(self,observed,enable_cf=True,enable_residual=True,enable_controller=True,
                enable_graph=None):
        """observed is a strict dict of x/mask/tod/dow only.

        Extra keys are rejected: the model cannot accidentally consume x_full/y.
        """
        if self.prediction_mode=='normal_anchored':
            return self.forward_normal_anchored(observed,enable_cf=enable_cf,
                 enable_controller=enable_controller,enable_graph=enable_graph)
        required={'x','mask','tod','dow'}
        if set(observed)!=required:
            raise ValueError(f'Only observed inputs allowed: {set(observed)-required} extra; {required-set(observed)} missing')
        x=observed['x'];mask=observed['mask'];tod=observed['tod'];dow=observed['dow']
        h0,context,p0=base_features(self.stla,x,mask,tod,dow)
        failure=self.failure(x,mask)
        if enable_graph is None:enable_graph=self.use_graph
        if enable_cf:
            temporal_cf=self.cf.temporal_tokens(h0,p0,failure)
            spatial_cf=(self.graph_cf(temporal_cf,failure['obs_reliability'])
                        if enable_graph else self._no_graph(temporal_cf))
            cf=self.cf(h0,p0,failure,spatial_cf,temporal=temporal_cf)
            cf_shift=cf['delta_h']+self.input_adapter(failure['tokens'])*0.1
            gate_cf=cf['latent_gate']
            h1=h0+gate_cf*cf_shift
            p1_candidate=decode(self.stla,h1,context)
            acceptance_logit,acceptance=self.cf.acceptance(cf,p0,p1_candidate)
            if not self.cf_error_gate:
                acceptance=torch.ones_like(acceptance)
            # Learned deployment policy: compare ACTUAL masked-only candidate to
            # protected baseline; no access to complete history or future Y.
            p1=p0+acceptance*(p1_candidate-p0)
        else:
            h1=h0;p1=p0;p1_candidate=p0;cf=None;gate_cf=None
            acceptance_logit=None;acceptance=torch.ones_like(p0)
        if enable_residual:
            # Residual must condition on *actually applied* CF correction, never ignored p_cf.
            if cf is not None:
                cf_tokens=cf['tokens']
            else:
                cf_tokens=x.new_zeros((x.shape[0],p0.shape[1],x.shape[2],self.cf.pos.embedding_dim))
            temporal_r=self.residual.temporal_tokens(h1,p0,p1,cf_tokens,failure)
            spatial_r=(self.graph_residual(temporal_r,failure['obs_reliability'])
                       if enable_graph else self._no_graph(temporal_r))
            res=self.residual(h1,p0,p1,cf_tokens,failure,spatial_r,temporal=temporal_r)
            h2=h1+res['delta_h']
            # IMPORTANT: take the residual branch's *incremental* effect relative
            # to the same RAW CF latent reference. Do not subtract the accepted
            # output P1 here: that would reintroduce CF correction rejected by
            # the output-space error-benefit gate.
            p2_shifted=decode(self.stla,h2,context)
            p2=p1+(p2_shifted-p1_candidate)
        else:
            p2=p1;res=None
        if enable_residual and enable_controller:
            if cf is None:
                # Residual-only ablations do not require CF controller features.
                cf_features={'tokens':cf_tokens,'spatial':self._no_graph(cf_tokens)}
            else:cf_features=cf
            control=self.controller(p0,p1,p2,cf_features,res,failure)
            gate_r=control['gate']
            pred=p1+gate_r*(p2-p1)
        else:
            control=None;gate_r=torch.ones_like(p0);pred=p2
        return {'prediction':pred,'backbone':p0,'cf_prediction':p1,
                'cf_residual_prediction':p2,'delta_cf':p1-p0,'delta_residual':p2-p1,
                'cf':cf,'residual':res,'gate_cf':gate_cf,'gate_residual':gate_r,
                'cf_raw_prediction':p1_candidate,
                'delta_cf_candidate':p1_candidate-p0,
                'cf_accept_logit':acceptance_logit,
                'cf_accept_gate':acceptance,
                'cf_teacher_gain':cf['teacher_gain'] if cf is not None else None,
                'control':control,'h0':h0,'cf_latent_delta':(h1-h0)}

    def _normal_core(self,x,mask,tod,dow,enable_graph,enable_cf):
        """Normal anchor + (optionally) the full-dose CF candidate for ONE view.

        Pure function of the masked inputs; identical arithmetic to the v0.2.2
        forward so that disabling the v0.2.3 mechanisms changes nothing.
        """
        h0,context,p0=base_features(self.stla,x,mask,tod,dow)
        failure=self.failure(x,mask)
        B,P,N=p0.shape
        blank_tokens=p0.new_zeros((B,P,N,self.cf.pos.embedding_dim))
        temporal_n=self.residual.temporal_tokens(h0,p0,p0,blank_tokens,failure)
        spatial_n=(self.graph_residual(temporal_n,failure['obs_reliability'])
                   if enable_graph else self._no_graph(temporal_n))
        normal=self.residual(h0,p0,p0,blank_tokens,failure,spatial_n,
                             temporal=temporal_n)
        p_normal=decode(self.stla,h0+normal['delta_h'],context)
        # During CF training and control, the Normal Anchor is literally fixed.
        # It also cannot be altered indirectly through shared failure features.
        if self.stage in ('anchor_cf','anchor_control'):
            p_normal=p_normal.detach()
        core={'h0':h0,'context':context,'p0':p0,'failure':failure,
              'normal':normal,'p_normal':p_normal,'cf':None}
        if enable_cf:
            temporal_cf=self.cf.temporal_tokens(h0,p0,failure)
            spatial_cf=(self.graph_cf(temporal_cf,failure['obs_reliability'])
                        if enable_graph else self._no_graph(temporal_cf))
            cf=self.cf(h0,p0,failure,spatial_cf,temporal=temporal_cf)
            raw_shift=(cf['delta_h']+0.1*self.input_adapter(failure['tokens']))
            applied_shift=cf['latent_gate']*raw_shift
            p_cf_raw=decode(self.stla,h0+applied_shift,context)
            core.update(cf=cf,applied_shift=applied_shift,p_cf_raw=p_cf_raw,
                        delta_cf=p_cf_raw-p0)
        return core

    def _view_consistency(self,x,mask,tod,dow,core,enable_graph):
        """Mechanism A: label-free reliability from K further-degraded views.

        Each view drops `consistency_drop` of the CURRENTLY OBSERVED entries.
        Views are computed without gradient; the features are detached.  At
        evaluation the drop pattern comes from a fixed seed, so results are
        deterministic; in training the global RNG is used.
        """
        K=self.consistency_views
        with torch.no_grad():
            shape=(K,)+tuple(mask.shape)
            if self.training:
                u=torch.rand(shape,device=mask.device)
            else:
                gen=torch.Generator(device=mask.device)
                gen.manual_seed(self.consistency_seed)
                u=torch.rand(shape,device=mask.device,generator=gen)
            keep=(u>=self.consistency_drop).to(mask.dtype)
            xk=(x[None]*keep).reshape((K*x.shape[0],)+tuple(x.shape[1:]))
            mk=(mask[None]*keep).reshape((K*mask.shape[0],)+tuple(mask.shape[1:]))
            rep=lambda t:t.repeat((K,)+(1,)*(t.dim()-1))
            view=self._normal_core(xk,mk,rep(tod),rep(dow),enable_graph,True)
            B=x.shape[0]
            basek=view['p_normal'].float().reshape((K,B)+tuple(view['p_normal'].shape[1:]))
            deltak=view['delta_cf'].float().reshape((K,B)+tuple(view['delta_cf'].shape[1:]))
            return consistency_features(core['p_normal'],core['delta_cf'],basek,deltak)

    def forward_normal_anchored(self,observed,enable_cf=True,enable_controller=True,
                                enable_graph=None):
        """Missing-input inference: Normal residual first, then mask-conditioned CF.

        Normal prediction P_N is an inference-time baseline, not oracle P_full.
        All train-only oracle losses are computed OUTSIDE this module.
        """
        required={'x','mask','tod','dow'}
        if set(observed)!=required:
            raise ValueError(f'Only observed inputs allowed: {set(observed)-required} extra; {required-set(observed)} missing')
        x=observed['x'];mask=observed['mask'];tod=observed['tod'];dow=observed['dow']
        if enable_graph is None:enable_graph=self.use_graph
        core=self._normal_core(x,mask,tod,dow,enable_graph,enable_cf)
        h0=core['h0'];p0=core['p0'];p_normal=core['p_normal'];normal=core['normal']
        cf=core['cf'];consistency=None;dose=None
        pred_gain=None;accept_logit=None
        if enable_cf:
            applied_shift=core['applied_shift'];p_cf_raw=core['p_cf_raw']
            delta_cf=core['delta_cf']
            p_candidate=p_normal+delta_cf
            pred_gain,accept_logit,accept_gate=self.cf.anchor_action(cf,p_normal,p_candidate)
            if not enable_controller:accept_gate=torch.ones_like(p_candidate)
            pred=p_normal+accept_gate*delta_cf
            if self.dose_enabled:
                if self.consistency_views>0:
                    consistency=self._view_consistency(x,mask,tod,dow,core,enable_graph)
                gamma=self.cf.dose_gamma(cf['tokens'])
                psi=dose_psi(self.cf.dose_grid,gamma)
                gains=self.cf.dose_gain(cf['tokens'],delta_cf,p_normal,psi,consistency)
                if enable_controller:
                    soft=self.training and self.stage=='anchor_control' and self.dose_soft_training
                    psi_sel,level_sel=select_dose(
                        gains.float(),psi.float(),self.cf.dose_grid,self.dose_margin,
                        temperature=(self.dose_temperature if soft else None))
                    accept_gate=psi_sel.to(p_candidate.dtype)
                    pred=p_normal+accept_gate*delta_cf
                else:
                    level_sel=None
                dose={'gamma':gamma,'psi':psi,'gain':gains,'selected_level':level_sel}
        else:
            applied_shift=torch.zeros_like(h0)
            p_cf_raw=p0
            delta_cf=torch.zeros_like(p0)
            p_candidate=p_normal
            accept_gate=torch.ones_like(p0)
            pred=p_normal
        return {
            'prediction':pred,'backbone':p0,'normal_prediction':p_normal,
            'normal_delta':p_normal-p0,'normal_cf_candidate':p_candidate,
            'cf_prediction':p_candidate,'cf_residual_prediction':p_candidate,
            'delta_cf':delta_cf,'delta_cf_candidate':delta_cf,
            'cf_raw_prediction':p_cf_raw,'cf':cf,'normal':normal,
            'residual':normal,'h0':h0,'cf_latent_delta':applied_shift,
            'cf_teacher_gain':cf['teacher_gain'] if cf is not None else None,
            'anchor_gain_pred':pred_gain,'cf_accept_logit':accept_logit,
            'cf_accept_gate':accept_gate,'gate_cf':accept_gate,
            'gate_residual':torch.ones_like(p0),
            'delta_residual':p_normal-p0,
            'dose_gamma':dose['gamma'] if dose else None,
            'dose_psi':dose['psi'] if dose else None,
            'dose_gain_pred':dose['gain'] if dose else None,
            'dose_selected_level':dose['selected_level'] if dose else None,
            'consistency':consistency,
        }
