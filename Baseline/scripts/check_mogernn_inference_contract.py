#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

GADI=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI if GADI.exists() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from baseline.runners.common import require_gpu, set_seed, validate_stage
from baseline.runners.mogernn_runner import (
    MoGERNNDataset, _prepare_x, assert_eval_target_independence, build_model,
)

ap=argparse.ArgumentParser()
ap.add_argument('--stage',choices=['development','formal'],default='development')
ap.add_argument('--seed',type=int,default=10)
a=ap.parse_args()

cfg,cache=validate_stage(a.stage,'MoGERNN')
set_seed(a.seed)
device=require_gpu()
c=cfg['runners']['MoGERNN']
ds=MoGERNNDataset(cache,'val')
dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
N=int(ds.x.shape[2])
adj=torch.from_numpy(np.asarray(np.load(cache/'adjacency.npy'),np.float32)).to(device)
if tuple(adj.shape)!=(N,N): raise RuntimeError(f'adjacency shape {tuple(adj.shape)} != {(N,N)}')
model=build_model(cfg,device)
x,obs,y,_v=next(iter(dl))
eps=float(c.get('epsilon_for_observed_zero',1e-6))
x=_prepare_x(x.float().to(device),obs.to(device),eps)
y=y.float().to(device).unsqueeze(-1)
res=assert_eval_target_independence(
    model,adj,x,y,0,device,float(c.get('eval_target_independence_atol',1e-6))
)
print('MOGERNN INFERENCE CONTRACT PASS',res)
