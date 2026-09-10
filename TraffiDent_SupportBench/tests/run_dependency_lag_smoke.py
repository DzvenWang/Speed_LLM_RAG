#!/usr/bin/env python3
"""Verify that sparse correlation anchors do not change raw lag duration."""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from supportbench.dependency import build_train_predictive_dependency


def main():
    rng=np.random.default_rng(20260905)
    T=1200; N=2; C=1
    src=rng.normal(0,1,T).astype(np.float32)
    target=np.zeros(T,np.float32)
    target[:2]=rng.normal(0,1,2)
    target[2:]=src[:-2] + rng.normal(0,0.01,T-2)
    traffic=np.zeros((T,N,C),np.float32)
    traffic[:,0,0]=src; traffic[:,1,0]=target
    valid=np.ones_like(traffic,dtype=bool)

    up_ids=np.full((N,1),-1,np.int64); up_w=np.zeros((N,1),np.float32)
    down_ids=np.full((N,1),-1,np.int64); down_w=np.zeros((N,1),np.float32)
    up_ids[1,0]=0; up_w[1,0]=1.0

    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'dep.npz'
        meta=build_train_predictive_dependency(
            traffic,valid,0,0,T,up_ids,up_w,down_ids,down_w,out,
            max_neighbors=1,max_lag_steps=3,sample_stride=12,min_pairs=30,
            corr_floor=0.0,graph_weight_mix=0.0,freq_minutes=5,
        )
        z=np.load(out)
        lag=int(z['lags'][1,0]); lag_min=int(z['lag_minutes'][1,0])
        assert lag==2, (lag,z['corr'][1,0])
        assert lag_min==10, lag_min
        assert int(z['sample_stride_minutes'])==60
        assert int(z['lag_step_minutes'])==5
        assert meta['tested_lag_minutes']==[0,5,10,15]
        print('DEPENDENCY LAG SMOKE PASS',{'lag_steps':lag,'lag_minutes':lag_min,'anchor_stride_minutes':int(z['sample_stride_minutes'])})


if __name__=='__main__':
    main()
