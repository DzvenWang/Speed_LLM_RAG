#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from backbone.metrics import _metric, per_query_metrics
p=np.array([[1.,3.]])
y=np.array([[0.,0.]])
m=np.array([[1,1]],bool)
assert abs(_metric(p,y,m,'mae')-2.0)<1e-12
assert abs(_metric(p,y,m,'rmse')-(5.0**0.5))<1e-12
pred=np.array([[[1.,3.]]]); true=np.zeros_like(pred); valid=np.ones_like(pred,dtype=bool)
direct=np.ones((1,2),dtype=bool); support=np.ones((1,2),dtype=bool)
r=per_query_metrics(pred,true,valid,direct,support)[0]
assert r['RMSE'] != r['MAE']
print('BACKBONE METRICS SMOKE PASS')
