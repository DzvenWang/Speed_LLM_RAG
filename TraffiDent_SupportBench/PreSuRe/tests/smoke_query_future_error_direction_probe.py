#!/usr/bin/env python3
from pathlib import Path
import sys, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.query_error_direction import QueryFutureErrorDirectionNet, derive_abs_error_thresholds, query_direction_loss

B,N,H,P,F=3,7,12,12,4*12+12+8+4+6
m=QueryFutureErrorDirectionNet(F,N,P,base_start=4*H,sense_dim=6,encoder_hidden=32,state_dim=16,node_dim=8,horizon_dim=6,hidden=24,dropout=0.0)
x=torch.randn(B,N,F)
em,dl=m(x)
assert em.shape==(B,P,N) and dl.shape==(B,P,N)
e=np.random.default_rng(1).normal(size=(8,P,N)).astype(np.float32); v=np.ones_like(e,dtype=bool)
th=derive_abs_error_thresholds(e,v,(.5,.75,.9)); assert set(th)=={'q50','q75','q90'} and len(th['q50'])==P
err=torch.tensor(e[:B]); valid=torch.ones_like(err,dtype=torch.bool); q50=torch.tensor(np.asarray(th['q50'],np.float32)[None,:,None])
loss,parts=query_direction_loss(em,dl,err,valid,1.0,q50,torch.tensor(1.0)); assert torch.isfinite(loss) and parts['eligible_fraction']>0
loss.backward(); assert any(p.grad is not None for p in m.parameters())
print('SMOKE QueryFutureErrorDirectionProbe v0.9.4 PASS')
