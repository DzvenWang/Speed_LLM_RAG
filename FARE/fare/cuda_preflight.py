"""Fail-fast CUDA FP16 integration gate using the actual frozen backbone.

Run *before* constructing expensive formal Memory or starting any epoch.
The temporary Memory is intentionally only for computational verification;
it never leaves RAM and its synthetic residuals are never used by training.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from .memory import EpisodeMemory
from .model import FARE
from .protocol import device_batch
from .training import forecasting_loss


def run_forward_backward_check(model,memory,batch,cfg,device):
    """One real AMP forward/backward, with explicit finite-value checks."""
    kind=torch.device(device).type
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=kind,dtype=torch.float16,enabled=True):
        result=model(batch,memory,purge=cfg['memory'].get('purge_steps',0),return_details=False)
        loss,_=forecasting_loss(result,batch,cfg)
    if not torch.isfinite(loss).all():
        raise RuntimeError('FARE CUDA AMP precheck: nonfinite loss')
    if not torch.isfinite(result['prediction']).all():
        raise RuntimeError('FARE CUDA AMP precheck: nonfinite prediction')
    loss.backward()
    checked=0
    for name,p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f'FARE CUDA AMP precheck: nonfinite gradient {name}')
            checked+=1
    if not checked:
        raise RuntimeError('FARE CUDA AMP precheck: no trainable gradient detected')
    if kind=='cuda':torch.cuda.synchronize(device)
    return {'loss':float(loss.detach()),'gradient_tensors':checked,
            'nodes':int(batch['x'].shape[-1]),'batch':int(batch['x'].shape[0])}


def run_cuda_preflight(backbone,graph,train,cfg,device,force_cpu=False):
    """Minimal actual vendor+data+FARE FP16 run; no persistent data or weights."""
    if torch.device(device).type!='cuda' and not force_cpu:return None
    if len(train)<3:raise RuntimeError('CUDA preflight requires at least three Train windows')
    # Dataset selected index order is not guaranteed chronological; sort by origin.
    if hasattr(train,'meta') and hasattr(train,'ix'):
        starts=train.meta['window_start'].to_numpy(np.int64)[train.ix]
        ordered=np.argsort(starts,kind='stable')
    else:
        starts=np.array([int(train[i]['origin'])-train.h+1 for i in range(len(train))],dtype=np.int64)
        ordered=np.argsort(starts,kind='stable')
    # Enough legal episodes for Top-16 + Top-8 and set replacement paths.
    e_count=min(40,len(train)-2)
    ep=[train[int(i)] for i in ordered[:e_count]]
    q=[train[int(i)] for i in ordered[-min(2,len(train)-e_count):]]
    x=torch.stack([item['x'] for item in ep]);mask=torch.stack([item['mask'] for item in ep])
    future=torch.stack([item['target'] for item in ep]);valid=torch.stack([item['future_valid'] for item in ep])
    start=torch.tensor([int(item['origin'])-train.h+1 for item in ep],dtype=torch.long)
    origin=torch.stack([item['origin'] for item in ep])
    # The approximation below is a TEST-ONLY source for shape/AMP/gradient checks,
    # never a scientifically valid residual or a saved Memory artifact.
    temp_mem=EpisodeMemory(x,mask,future,valid,start,origin,base_forecast=future*.9)
    temp_mem.attach_graph(graph)
    batch=device_batch(default_collate(q),device)
    if not temp_mem.valid_for(batch['origin'],device).any(-1).all():
        raise RuntimeError('CUDA preflight: no chronologically lawful historical evidence')
    print('FARE4_CUDA_AMP_PRECHECK_START',flush=True)
    # Avoid changing the RNG trajectory of real training model initialization.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.device(device).type=='cuda' else []):
        model=FARE(backbone,graph,cfg).to(device)
        try:
            report=run_forward_backward_check(model,temp_mem,batch,cfg,device)
        finally:
            model.zero_grad(set_to_none=True)
            del model
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    print('FARE4_CUDA_AMP_PRECHECK_PASS '+json.dumps(report),flush=True)
    return report
