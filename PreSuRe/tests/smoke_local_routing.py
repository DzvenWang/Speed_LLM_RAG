#!/usr/bin/env python3
from pathlib import Path
import importlib.util,sys,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location('rv',R/'scripts/run_variant.py');rv=importlib.util.module_from_spec(spec);spec.loader.exec_module(rv)
from presure.models import HorizonUtilityNet,RegionHorizonUtilityNet,LocalSafeGate
B,K,P,N,D=3,6,4,10,12;regions=np.array([0,0,0,1,1,1,2,2,2,2],np.int32);Rr=3
rng=np.random.default_rng(3);base=rng.normal(size=(B,P,N)).astype('f4');y=rng.normal(size=(B,P,N)).astype('f4');valid=np.ones((B,P,N),bool);direct=rng.random((B,N))>.5;support=rng.random((B,N))>.6;obs=rng.random((B,5,N))>.2;rr=rng.normal(size=(B,K,P,N)).astype('f4')*.2
cfg={'oracle':{'objective_weights':{'overall':.45,'masked':.35,'support':.2},'alpha_grid':[.25,.5,.75,1]},'local_routing':{'m3_top_k':3,'m4_top_k':3,'utility_candidate_k':6},'model':{'m3':{'positive_utility_epsilon':0},'m4':{'positive_utility_epsilon':0}}}
hlocal=rv.horizon_local_features(base,obs,direct,support,rr);rlocal=rv.region_horizon_local_features(base,obs,direct,support,rr,regions);assert hlocal.shape==(B,K,P,8);assert rlocal.shape==(B,K,P,Rr,8)
ht=rv.horizon_utility_targets(base,y,valid,direct,support,rr,cfg);rt=rv.region_horizon_utility_targets(base,y,valid,direct,support,rr,regions,cfg);assert ht.shape==(B,K,P);assert rt.shape==(B,K,P,Rr)
q=torch.randn(B,K,D);c=torch.randn(B,K,D);sim=torch.randn(B,K);rrt=torch.tensor(rr)
hn=HorizonUtilityNet(D,P);hu=hn(q,c,sim,torch.tensor(hlocal));hc,ha=rv.horizon_corr_from_utility(hu,rrt,3);assert hc.shape==(B,P,N) and ha.shape==(B,P)
rn=RegionHorizonUtilityNet(D,P,Rr);ru=rn(q,c,sim,torch.tensor(rlocal));rc,ra=rv.region_corr_from_utility(ru,rrt,regions,3);assert rc.shape==(B,P,N) and ra.shape==(B,P,Rr)
g=LocalSafeGate(D,P,N);go=g(torch.randn(B,D),torch.tensor(base),rc,torch.tensor(obs.mean(1),dtype=torch.float32),torch.tensor(direct),torch.tensor(support));assert go.shape==(B,P,N);assert torch.all((go>=0)&(go<=1))
print('PRESURE LOCAL ROUTING SMOKE PASS',{'horizon_utility':tuple(hu.shape),'region_utility':tuple(ru.shape),'gate':tuple(go.shape)})
