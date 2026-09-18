#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from presure.models import PredictiveUtilityRerankerV09A
from presure.predictive_utility import (
    calibrate_abstain_threshold,
    node_horizon_utility_targets,
    predictive_utility_safe_loss,
    safe_policy_metrics_from_records,
    safe_selection_score,
)

cfg = yaml.safe_load((R / 'configs/presure.yaml').read_text())
assert cfg['version'] in {'0.9.1','0.9.2','0.9.3','0.9.4'}
B,N,K,P,F = 2,7,5,4,31
m = PredictiveUtilityRerankerV09A(
    node_feature_dim=F, num_nodes=N, P=P, encoder_hidden=24, state_dim=12,
    node_dim=5, sense_dim=6, residual_dim=8, pair_dim=16, hidden=24,
    horizon_dim=4, attention_heads=4, dropout=0.0,
)
q = torch.randn(B,N,F); s = torch.randn(B,N,K,F)
qs=q[...,-6:]; ss=s[...,-6:]
rr=torch.randn(B,N,K,P)*0.2; qb=torch.randn(B,P,N); sb=torch.randn(B,N,K,P); coarse=torch.randn(B,K)
mu,sigma,blogit=m(q,s,qs,ss,rr,qb,sb,coarse)
cm=torch.ones(B,N,K,dtype=torch.bool); valid=torch.ones(B,N,P,dtype=torch.bool)
true_need=torch.randn(B,P,N)
target,_=node_horizon_utility_targets(true_need,rr,cm,[0.25,0.5,1.0],0.1,1.5)
loss,parts=predictive_utility_safe_loss(mu,sigma,blogit,target,cm,valid,coarse,cfg)
assert torch.isfinite(loss); loss.backward()
assert parts['top1_n'] >= 0
score=safe_selection_score(mu,sigma,blogit,0.5,0.25)
assert score.shape==(B,N,K,P) and torch.isfinite(score).all()

# Calibrator must respect the explicit coverage floor and use no future inputs at deployment.
rng=np.random.default_rng(10)
top_score=np.linspace(-1,1,1000).astype(np.float32)
chosen=(0.08*top_score + rng.normal(0,0.03,1000)).astype(np.float32)
best=np.maximum(chosen, rng.uniform(0.02,0.25,1000)).astype(np.float32)
choice,grid=calibrate_abstain_threshold(top_score,chosen,best,cfg)
assert choice['coverage'] + 1e-6 >= cfg['predictive_utility_safe_selection']['policy']['min_coverage']
assert len(grid)==len(cfg['predictive_utility_safe_selection']['policy']['coverage_grid'])
metrics=safe_policy_metrics_from_records(top_score,chosen,best,choice['threshold'],0.02,0.02)
assert 0 <= metrics['coverage'] <= 1
assert metrics['mean_safe_policy_regret'] >= -1e-8
print('SMOKE PredictiveUtilitySafeSelection v0.9A.1 PASS')
