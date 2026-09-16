#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from presure.utils import load_yaml
from presure.proposal_meta import (
    SupportProposalMetaRanker, build_support_proposals,
    meta_ranker_loss, ProposalMetricAccumulator,
)

cfg=load_yaml(R/'configs/presure.yaml')
assert cfg['version']=='0.7.6'
order=['generic','hard','pred10','pred20']
B=N=1;L=4;P=2
# Four experts propose complementary supports; pred10 duplicates generic on h0 only.
scores={
 'generic':torch.tensor([[[[4.,4.],[2.,2.],[1.,1.],[0.,0.]]]],dtype=torch.float32),
 'hard':torch.tensor([[[[0.,0.],[4.,4.],[2.,2.],[1.,1.]]]],dtype=torch.float32),
 'pred10':torch.tensor([[[[3.,0.],[1.,4.],[2.,2.],[0.,1.]]]],dtype=torch.float32),
 'pred20':torch.tensor([[[[0.,0.],[1.,1.],[2.,2.],[4.,4.]]]],dtype=torch.float32),
}
target=torch.tensor([[[[0.8,-0.2],[0.3,0.7],[-0.4,0.1],[-0.6,-0.5]]]],dtype=torch.float32)
rr=torch.tensor([[[[1.,-1.],[2.,2.],[-1.,1.],[-2.,-2.]]]],dtype=torch.float32)
conf=torch.tensor([[[.9,.8,.7,.6]]],dtype=torch.float32)
cm=torch.ones((B,N,L),dtype=torch.bool)
need=torch.tensor([[[.9],[.8]]],dtype=torch.float32)  # [B,P,N]
b=build_support_proposals(scores,order,target,rr,conf,cm,need,0.2,2.0)
assert b['features'].shape[:4]==(B,N,P,4)
assert b['target'].shape==(B,N,P,4)
assert b['candidate_idx'].shape==(B,N,P,4)
assert int(b['mask'].sum())<B*N*P*4  # duplicate proposal really merged

m=SupportProposalMetaRanker(b['features'].shape[-1],32,0.0)
out=m(b['features'])
valid=torch.ones((B,P,N),dtype=torch.bool)
loss,parts=meta_ranker_loss(out,b['target'],b['mask'],valid,need,cfg)
assert torch.isfinite(loss) and parts['cells']>=1
loss.backward()
assert any(p.grad is not None for p in m.parameters())

# Oracle scoring must have zero regret and never select a worse proposal.
a=ProposalMetricAccumulator(0.02,0.02,128,10)
a.update(b['target'],b['target'],b['mask'],valid.permute(0,2,1))
s=a.summary()
assert s['mean_selection_regret']<1e-8
assert s['near_oracle_rate']==1.0
assert s['selected_harmful_rate']<1.0
print('PRESURE V0.7.6 REGRET-AWARE SUPPORT PROPOSAL META-RANKING PROBE SMOKE PASS')
