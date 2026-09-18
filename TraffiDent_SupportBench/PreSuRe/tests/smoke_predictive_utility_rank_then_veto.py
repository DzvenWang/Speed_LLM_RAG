from pathlib import Path
import sys, yaml, torch
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R))
from presure.models import PredictiveUtilityRerankerV09A, StrongHarmVetoHead
from presure.harm_veto import strong_harm_focal_loss, sequential_veto_choice, frozen_rank_order

cfg=yaml.safe_load((R/'configs/presure.yaml').read_text()); assert cfg['version'] in {'0.9.2','0.9.3','0.9.4'}
B,N,K,P,F=2,5,6,4,23
m=PredictiveUtilityRerankerV09A(F,N,P,encoder_hidden=24,state_dim=8,node_dim=4,sense_dim=6,residual_dim=12,pair_dim=16,hidden=24,horizon_dim=4,attention_heads=4,dropout=0.0)
q=torch.randn(B,N,F); s=torch.randn(B,N,K,F); qs=torch.randn(B,N,6); ss=torch.randn(B,N,K,6); rr=torch.randn(B,N,K,P); qb=torch.randn(B,P,N); sb=torch.randn(B,N,K,P); cs=torch.randn(B,K)
with torch.no_grad(): role,mu,sigma,blogit=m.forward_features(q,s,qs,ss,rr,qb,sb,cs); r=mu-.5*sigma
v=StrongHarmVetoHead(role_dim=96,hidden=32,dropout=0.0); h=v(role,mu,sigma,blogit,r)
target=torch.randn(B,N,K,P)*.1; cm=torch.ones(B,N,K,dtype=torch.bool); valid=torch.ones(B,N,P,dtype=torch.bool)
loss,parts=strong_harm_focal_loss(h,target,cm,valid,.02,rank_score=r,top_rank_weight=2.0,top_rank_k=3); loss.backward(); assert torch.isfinite(loss)
order=frozen_rank_order(r,cm,valid); assert order.shape==(B,N,K,P)
choice,acc,checks,_=sequential_veto_choice(r,torch.sigmoid(h.detach()),cm,valid,.5,4,order=order); assert choice.shape==(B,N,P) and acc.shape==(B,N,P) and checks.max()<=4
# Regression guard for the 179261672 calibration/final-eval NameError: both
# policy paths must explicitly retain rscore returned by model_outputs.
runner=(R/'scripts/run_predictive_utility_rank_then_veto_probe.py').read_text()
assert runner.count('role,mu,sigma,blogit,rscore,hlogit,cm,valid,target=model_outputs') >= 2
assert runner.count('frozen_rank_order(rscore,cm,') >= 2
# ranker remains unchanged by veto backward
assert all(p.grad is None for p in m.parameters())
print('SMOKE PredictiveUtilityRankThenVeto v0.9A.2 PASS')
