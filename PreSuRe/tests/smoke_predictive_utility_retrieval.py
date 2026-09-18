#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from presure.models import PredictiveUtilityRerankerV09A
from presure.predictive_utility import node_horizon_utility_targets, predictive_utility_loss, risk_score
from presure.need_gated import gather_support_context

cfg = yaml.safe_load((R / 'configs/presure.yaml').read_text())
assert cfg['version'] in {'0.9.0','0.9.1','0.9.2','0.9.3','0.9.4','0.9.5','0.9.5.1'}
B,N,K,P,F = 2,7,5,4,31
m = PredictiveUtilityRerankerV09A(
    node_feature_dim=F, num_nodes=N, P=P, encoder_hidden=24, state_dim=12,
    node_dim=5, sense_dim=6, residual_dim=8, pair_dim=16, hidden=24,
    horizon_dim=4, attention_heads=4, dropout=0.0,
)
q = torch.randn(B,N,F)
s = torch.randn(B,N,K,F)
qs = q[...,-6:]
ss = s[...,-6:]
rr = torch.randn(B,N,K,P) * 0.2
qb = torch.randn(B,P,N)
sb = torch.randn(B,N,K,P)
coarse = torch.randn(B,K)
mu,sigma,blogit = m(q,s,qs,ss,rr,qb,sb,coarse)
assert mu.shape == sigma.shape == blogit.shape == (B,N,K,P)
assert torch.all(sigma > 0)
true_need = torch.randn(B,P,N)
cm = torch.ones(B,N,K,dtype=torch.bool)
valid = torch.ones(B,N,P,dtype=torch.bool)
target,alpha = node_horizon_utility_targets(true_need,rr,cm,[0.25,0.5,1.0],0.1,1.5)
assert target.shape == alpha.shape == (B,N,K,P)
# Use real config training block but synthetic tensors.
loss,parts = predictive_utility_loss(mu,sigma,blogit,target,cm,valid,coarse,cfg)
assert torch.isfinite(loss)
loss.backward()
assert torch.isfinite(risk_score(mu,sigma,0.5)).all()

# Regression test for the v0.9A episode-level [B,K] candidate gather contract.
T = 11
gids2 = np.asarray([[1,2,3,4,5],[2,3,4,5,6]], dtype=np.int64)
node_mem = np.random.randn(T,N,F).astype(np.float32)
res_mem = np.random.randn(T,P,N).astype(np.float32)
gids_node = np.broadcast_to(gids2[:,None,:], (B,N,K))
ctx2, rr2 = gather_support_context(node_mem, res_mem, gids_node)
assert ctx2.shape == (B,N,K,F)
assert rr2.shape == (B,N,K,P)

print('SMOKE PredictiveUtilityRetrieval v0.9A PASS')
