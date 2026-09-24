#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from presure.utils import load_yaml
from presure.utility_specialist import true_q90_mask,specialist_loss,two_expert_score,CandidateMetricAccumulator

cfg=load_yaml(R/'configs/presure.yaml')
assert cfg['version'] in {'0.7.4','0.7.5','0.7.6','0.8.0','0.9.0','0.9.1','0.9.2','0.9.2','0.9.2','0.9.2','0.9.2','0.9.3','0.9.4'}
assert cfg['need_conditioned_utility_specialist_probe']['initialize_from_utility_a'] is True

# Horizon-specific q90 labels are supervision only.
base=np.zeros((2,2,1),np.float32); truth=np.array([[[0.5],[3.0]],[[2.0],[0.2]]],np.float32)
hard=true_q90_mask(base,truth,[1.0,2.0])
assert hard.shape==(2,2,1) and hard[0,1,0] and hard[1,0,0] and not hard[0,0,0]

# Hard specialist loss must prefer correct pair ordering and clear beneficial/harmful labels.
pred=torch.tensor([[[[1.0],[-1.0]]]],requires_grad=True) # B,N,L,P
target=torch.tensor([[[[0.8],[-0.8]]]])
cm=torch.tensor([[[True,True]]])
valid=torch.tensor([[[True]]]) # B,P,N
hm=torch.tensor([[[True]]])
need=torch.tensor([[[0.9]]])
loss,rank,cls,racc,rn=specialist_loss(pred,target,cm,valid,hm,need,cfg,'hard')
assert rn==1 and float(racc)==1.0 and torch.isfinite(loss)
loss.backward(); assert pred.grad is not None

# Need-weighted loss also runs without privileged hard-cell routing.
pred2=torch.tensor([[[[0.4],[-0.2]]]],requires_grad=True)
loss2,*_=specialist_loss(pred2,target,cm,valid,hm,need,cfg,'weighted')
assert torch.isfinite(loss2); loss2.backward()

# Two-expert route must use Generic below threshold and move toward Hard above threshold.
g=torch.tensor([[[[0.0,0.0],[0.0,0.0]]]],dtype=torch.float32) # [1,1,2,2]
h=torch.tensor([[[[2.0,2.0],[-2.0,-2.0]]]],dtype=torch.float32)
npb=torch.tensor([[[0.1],[0.9]]],dtype=torch.float32) # [B,P,N] = [1,2,1]
out=two_expert_score(g,h,npb,0.5,1.0,0.2)
assert out.shape==g.shape
assert abs(float(out[0,0,0,0])-0.5)<1e-6
assert float(out[0,0,0,1])>0.9

acc=CandidateMetricAccumulator(0.02,128,10)
cell=torch.tensor([[[True]]])
acc.update(target,target,cm,cell)
s=acc.summary(); assert s['pairwise_rank_accuracy']==1.0 and s['selected_beneficial_rate']==1.0
print('PRESURE V0.7.4 NEED-CONDITIONED UTILITY SPECIALIST PROBE SMOKE PASS')
