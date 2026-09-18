#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))

from presure.need_gated import (
    RankAccumulator, activation_diagnostics, mae_views, quantile_threshold,
    relative_gains, utility_targets,
)
from presure.utils import load_yaml

cfg=load_yaml(R/'configs/presure.yaml')
assert cfg['version'] in {'0.7.3','0.7.4','0.7.5','0.7.6','0.8.0','0.9.0','0.9.1','0.9.2','0.9.2','0.9.2','0.9.2','0.9.2','0.9.2','0.9.3','0.9.4','0.9.5','0.9.5.1'}
assert cfg['need_gated_utility_probe']['reuse_only'] is True

# Quantile gate must retain the highest-score tail.
s=np.arange(100,dtype=np.float32).reshape(10,2,5)
m=np.ones_like(s,dtype=bool)
th=quantile_threshold(s,m,0.10)
assert 88 <= th <= 90
assert np.mean(s>=th) >= 0.09

# Relative utility: a matching residual must beat an opposite residual.
need=torch.tensor([[[1.0]]])  # [B,P,N]
rr=torch.tensor([[[[1.0],[-1.0]]]], dtype=torch.float32)  # [1,1,2,1] B,N,L,P
cm=torch.tensor([[[True,True]]])
u=utility_targets(need,rr,cm,cfg)
assert u.shape==(1,1,2,1)
assert float(u[0,0,0,0]) > float(u[0,0,1,0])

# Ranking accumulator should recognize the correct support.
score=u.clone()
valid=torch.tensor([[[True]]])  # [B,N,P]
a=RankAccumulator(); a.update(score,u,cm,valid,0.02)
z=a.summary()
assert z['top1_agreement']==1.0
assert z['selected_beneficial_rate']==1.0

# Correction metrics are consistent.
base=np.zeros((2,1,2),np.float32)
y=np.ones_like(base)
gate=np.array([[[True,False]],[[False,True]]])
corr=base.copy(); corr[gate]=0.5
valid=np.ones_like(base,dtype=bool)
direct=np.ones((2,2),dtype=bool); support=np.ones((2,2),dtype=bool)
b=mae_views(base,y,valid,direct,support); c=mae_views(corr,y,valid,direct,support)
g=relative_gains(b,c)
assert 0 < g['masked'] < 1
ad=activation_diagnostics(base,y,corr,gate,valid,direct,support)
assert ad['overall']['activation_rate']==0.5
assert ad['overall']['harmful_fraction']==0.0

print('PRESURE V0.7.3 NEED-GATED UTILITY PROBE SMOKE PASS')
