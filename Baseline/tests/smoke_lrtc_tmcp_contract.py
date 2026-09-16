#!/usr/bin/env python3
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.compat.lrtc_tmcp import recover_batch

torch.manual_seed(10)
x=torch.rand(3,5,1,12)*60.0
mask=torch.ones_like(x,dtype=torch.bool)
mask[0,1:4,0,3:8]=False
mask[1,0:2,0,0:6]=False
mask[2,3:5,0,6:12]=False
sparse=torch.where(mask,x,torch.zeros_like(x))
y,info=recover_batch(sparse,mask,max_iter=3)
assert y.shape==x.shape
assert torch.isfinite(y).all()
assert torch.equal(y[mask],x[mask]), 'observed entries must be protected exactly'
assert info.iterations>=1
print('LRTC-TMCP CONTRACT SMOKE PASS', info)
