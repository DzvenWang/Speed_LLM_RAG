#!/usr/bin/env python3
from pathlib import Path
import importlib.util,sys,numpy as np,torch,yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from presure.probes import PredictiveNeedNet,SupportUtilityProbeNet,exact_spearman_sample,pearson,binary_auroc
spec=importlib.util.spec_from_file_location('probe',R/'scripts/run_predictive_need_probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
cfg=yaml.safe_load((R/'configs/presure.yaml').read_text())
# Small synthetic contract: H=3, P=4 -> base slice starts at 12.
B,N,L,H,P,F=2,7,3,3,4,36;rng=np.random.default_rng(17)
x=rng.normal(size=(B,N,F)).astype('f4');x[...,12:16]=rng.normal(size=(B,N,P));x[...,-6:]=rng.random((B,N,6))
need=PredictiveNeedNet(F,N,P,4*H,hidden=32,demb=16,node_dim=8,horizon_dim=4,dropout=0.0);pred=need(torch.tensor(x));assert pred.shape==(B,P,N)
target=torch.randn_like(pred);valid=torch.ones_like(pred,dtype=torch.bool);cfg2={'predictive_need_probe':{'need_model':{'huber_beta':.25,'target_weight_clip':3.,'magnitude_weight':.5,'sign_min_abs_norm':.1,'sign_weight':.1}}}
loss,reg,sl=probe.need_loss(pred,target,valid,cfg2);loss.backward();assert torch.isfinite(loss)
q=torch.tensor(x);s=torch.tensor(rng.normal(size=(B,N,L,F)).astype('f4'));s[...,-6:]=torch.rand(B,N,L,6);s[...,12:16]=torch.randn(B,N,L,P)
rr=torch.randn(B,N,L,P)*.2;qb=torch.randn(B,P,N);sb=torch.randn(B,N,L,P);conf=torch.softmax(torch.randn(B,N,L),-1);qs=q[...,-6:];ss=s[...,-6:]
util=SupportUtilityProbeNet(F,P,context_dim=16,residual_dim=8,pair_dim=16,hidden=32,horizon_dim=4,attention_heads=4,dropout=0.0)
score=util(q,s,qs,ss,rr,qb,sb,conf,pred.detach());assert score.shape==(B,N,L,P)
cm=torch.ones(B,N,L,dtype=torch.bool);ucfg={'predictive_need_probe':{'utility_model':{'alpha_grid':[.25,.5,1.0],'gain_floor':.1,'utility_clip':1.5,'huber_beta':.1,'beneficial_threshold':.02,'classification_temperature':.2,'rank_min_gap':.02,'rank_margin':.1,'rank_weight':.5,'classification_weight':.2}}}
ut=probe.utility_targets(target,rr,cm,ucfg);ul,*rest=probe.utility_loss(score,ut,cm,valid,ucfg);ul.backward();assert torch.isfinite(ul)
a=np.linspace(-1,1,100);b=a+rng.normal(scale=.05,size=100);assert exact_spearman_sample(a,b)>0.9 and pearson(a,b)>0.9 and binary_auroc(a,a>0)>.9
print('PRESURE PREDICTIVE NEED PROBE SMOKE PASS',{'need':tuple(pred.shape),'utility':tuple(score.shape),'need_loss':float(loss.detach()),'utility_loss':float(ul.detach())})

# Decision-gate contract smoke
needm={'overall':{'sign_accuracy_meaningful':0.7,'spearman_sample':0.3}}
util={'A':{'pairwise_rank_accuracy':0.55},'B':{'pairwise_rank_accuracy':0.63},'C':{'pairwise_rank_accuracy':0.8}}
ana={'pairwise_rank_accuracy':0.66}
status,reasons=probe.decision(needm,util,ana,cfg);assert status=='GO_V08' and reasons['analytic_predicted_need_utility_pass']
print('PRESURE PREDICTIVE NEED DECISION SMOKE PASS',status)
