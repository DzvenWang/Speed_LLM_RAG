#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from presure.utils import load_yaml
from presure.deployment_routing import (
    need_thresholds_by_horizon, predicted_need_mask, deployment_specialist_loss,
    ExpertPreferenceRouter, router_features, router_targets, router_loss,
    route_candidate_scores, oracle_expert_scores,
)

cfg=load_yaml(R/'configs/presure.yaml')
assert cfg['version'] in {'0.7.5','0.7.6','0.8.0','0.9.0','0.9.1','0.9.2','0.9.2','0.9.2','0.9.2','0.9.3','0.9.4'}

# Train-only horizon-specific predicted-Need thresholds.
p=np.array([[[0.1,0.9],[0.2,0.8]],[[0.3,0.7],[0.4,0.6]]],np.float32) # B=2,P=2,N=2
v=np.ones_like(p,dtype=bool)
th=need_thresholds_by_horizon(p,v,0.25)
m=predicted_need_mask(p,th,v)
assert th.shape==(2,) and m.shape==p.shape and int(m.sum())>=2

# Deployment specialist objective runs only on predicted-Need cells.
pred=torch.tensor([[[[1.0],[-1.0]]]],requires_grad=True)
target=torch.tensor([[[[0.8],[-0.8]]]])
cm=torch.tensor([[[True,True]]])
valid=torch.tensor([[[True]]]) # B,P,N
dep=torch.tensor([[[True]]])
loss,rank,cls,racc,rn=deployment_specialist_loss(pred,target,cm,valid,dep,cfg)
assert rn==1 and float(racc)==1.0 and torch.isfinite(loss)
loss.backward(); assert pred.grad is not None

# Router features are deployment-safe summaries of expert scores + Need + M3 confidence.
scores={
 'generic':torch.tensor([[[[2.0],[-1.0]]]],dtype=torch.float32),
 'hard':torch.tensor([[[[-1.0],[2.0]]]],dtype=torch.float32),
 'pred10':torch.tensor([[[[-0.5],[1.5]]]],dtype=torch.float32),
 'pred20':torch.tensor([[[[-0.2],[0.8]]]],dtype=torch.float32),
}
need=torch.tensor([[[0.9]]],dtype=torch.float32)
conf=torch.tensor([[[0.8,0.2]]],dtype=torch.float32)
feat,_=router_features(scores,['generic','hard','pred10','pred20'],need,conf,cm,0.2)
r=ExpertPreferenceRouter(4,32,0.0)
logits=r(feat)
assert logits.shape==(1,1,1,4)

# Future-grounded router label says Generic is best here because its selected utility is positive.
labels,use,u=router_targets(scores,['generic','hard','pred10','pred20'],target,cm,valid,0.01)
assert use.item() and labels.item() in range(4)
rl,ra,rn=router_loss(logits,labels,use,need,1.5)
assert rn==1 and torch.isfinite(rl)

hard=route_candidate_scores(torch.tensor([[[[10.,-10.,-10.,-10.]]]]),scores,['generic','hard','pred10','pred20'],0.2,False)
assert hard.shape==pred.shape and float(hard[0,0,0,0])>float(hard[0,0,1,0])
orc=oracle_expert_scores(scores,['generic','hard','pred10','pred20'],target,cm,0.2)
assert orc.shape==pred.shape
print('PRESURE V0.7.5 DEPLOYMENT-ALIGNED UTILITY ROUTER PROBE SMOKE PASS')
