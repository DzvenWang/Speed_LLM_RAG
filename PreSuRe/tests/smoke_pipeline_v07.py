#!/usr/bin/env python3
from pathlib import Path
import importlib.util,sys,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
spec=importlib.util.spec_from_file_location('rv',R/'scripts/run_variant.py');rv=importlib.util.module_from_spec(spec);spec.loader.exec_module(rv)
from presure.models import NodeUsefulnessNet,PairwiseSupportRouter
B,M,Q,K,P,N,D,F=2,20,5,6,4,7,8,38;rng=np.random.default_rng(7)
zmem=rng.normal(size=(M,N,D)).astype('f4');zq=rng.normal(size=(Q,N,D)).astype('f4')
nmem=rng.normal(size=(M,N,F)).astype('f4');nq=rng.normal(size=(Q,N,F)).astype('f4')
nmem[...,-6:]=rng.random((M,N,6));nq[...,-6:]=rng.random((Q,N,6))
# Make the node-context PatchSTG forecast slice (4H:4H+P) meaningful. H=5 => 20:24.
nmem[...,20:24]=rng.normal(size=(M,N,P)).astype('f4');nq[...,20:24]=rng.normal(size=(Q,N,P)).astype('f4')
res=rng.normal(size=(M,P,N)).astype('f4')*.4;base=(40+rng.normal(size=(Q,P,N))*4).astype('f4')
inds=np.vstack([rng.choice(M,K,replace=False) for _ in range(Q)]).astype('i4');scores=rng.normal(size=(Q,K)).astype('f4');cand={'indices':inds,'scores':scores}
cfg={'experiment':{'history':5,'horizon':P},'node_routing':{'utility_candidate_k':K,'m4_top_k':3},'model':{'m3':{'use_threshold':.45,'router_use_threshold':.20}}}
dev=torch.device('cpu');m3=NodeUsefulnessNet(D,max_scale=1.5);router=PairwiseSupportRouter(D,F,P,context_dim=12,residual_dim=8,pair_dim=16,hidden=32,horizon_dim=4,attention_heads=4)
ids=np.array([1,3]);raw,p,sc=rv.m3_raw_correction(m3,zmem,zq,nmem,nq,cand,res,ids,cfg,dev,True);assert raw.shape==(2,P,N)
gid,csel,cssel,conf,m3scale,cmask,rr,qctx,cctx=rv.m3_select_for_pairwise_router(m3,zmem,zq,nmem,nq,cand,res,ids,cfg,dev)
assert rr.shape==(2,N,3,P) and qctx.shape==(2,N,F) and cctx.shape==(2,N,3,F)
y=base+rng.normal(size=base.shape).astype('f4');valid=np.ones_like(base,dtype=bool);b=torch.tensor(base[ids]);yy=torch.tensor(y[ids]);vv=torch.tensor(valid[ids]);q=torch.tensor(zq[ids]);qs=torch.tensor(nq[ids,...,-6:])
corr,mass,dis,slog,scale,noop=rv._m4_forward_selected(router,q,qs,qctx,csel,cssel,cctx,conf,cmask,rr,b,40,10,cfg)
probs,tdelta,tcorr,tmass,hard,urel,ent=rv.m4_soft_teacher_targets(b,yy,vv,rr,cmask,[.25,.5,1.0],temperature=.12,mass_temperature=.05,min_abs_gain=.01,gain_floor=.1,utility_clip=1.5)
assert probs.shape==(2,N,4,P) and torch.allclose(probs.sum(2),torch.ones_like(tmass),atol=1e-5)
ce,dl,cl,mm,acc,te=rv.m4_soft_distill_loss(slog,scale,noop,rr,corr,mass,probs,tdelta,tcorr,tmass,vv)
rl,ra=rv.m4_pairwise_utility_rank_loss(slog,urel,cmask,vv,.1,.02)
for x in [ce,dl,cl,mm,acc,te,rl,ra]: assert torch.isfinite(x)
tgt,gain,relg=rv.continuous_gate_teacher(b,corr,yy,vv,.01,.1); fakeg=torch.sigmoid(torch.randn_like(b));gl=rv.continuous_gate_loss(fakeg,tgt,relg,vv,2.0);assert torch.isfinite(gl)
print('PRESURE V0.7 PIPELINE SMOKE PASS',{'selected':tuple(rr.shape),'query_ctx':tuple(qctx.shape),'support_ctx':tuple(cctx.shape),'m4_corr':tuple(corr.shape),'teacher_probs':tuple(probs.shape),'rank_acc':float(ra)})
