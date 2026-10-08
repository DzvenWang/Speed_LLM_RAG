#!/usr/bin/env python3
"""Real-cache Smoke gradient connectivity audit for FARE's major trainable paths.

The initial zero-initialized STLA/controller adapters can mask gradients on step
one; report both steps and never treat first-step zero as a fatal error.
"""
from __future__ import annotations
import argparse,json
import _bootstrap
import torch
from fare.config import load_config
from fare.runtime import setup
from fare.losses import losses
from fare.data import loader,to_device
from fare.trainer import seed_all

GROUPS={
 'observability':'observability.',
 'retrieval_query':'retrieval.encoder.observability_query.',
 'reranker':'retrieval.rerank.',
 'state_observability':'state_recovery.obs_health.',
 'state_recovery':'state_recovery.',
 'innovation':'innovation.',
 'reliability_observability':'reliability.obs_reliability.',
 'controller':'controller.',
 'stla_support':'stla.graph_adapter.',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--targets',nargs='+',choices=['speed','flow'],default=['speed','flow'])
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--steps',type=int,default=2)
    args=p.parse_args()
    if args.steps<2:raise ValueError('Use >=2 steps to unlock zero-initialized support adapters')
    for target in args.targets:
        cfg=load_config(dataset=args.dataset,target=target,seed=args.seed,smoke=True)
        seed_all(args.seed)
        m,mem,tr,va,device,meta=setup(cfg,smoke=True)
        m.set_phase('full');m.to(device);mem.to(device)
        # setup() creates Train-only memory, but it deliberately leaves the
        # learned retrieval index uninitialized. Build it BEFORE the first
        # forward, exactly as FARETrainer does at each training epoch.
        m.refresh(mem)
        if mem.keys is None or len(mem.keys)!=len(mem.x):
            raise RuntimeError('FARE_GRADIENT_INDEX_INIT_FAILED: invalid Train memory key index')
        print('FARE_GRADIENT_INDEX_READY '+json.dumps({
            'dataset':args.dataset,'target':target,
            'memory_rows':len(mem.x),'index_shape':list(mem.keys.shape)}),flush=True)
        opt=torch.optim.AdamW([par for par in m.parameters() if par.requires_grad],lr=1e-4)
        processed=0
        for step,batch in enumerate(loader(tr,2,shuffle=False,workers=0)):
            if step>=args.steps:break
            if step>0:m.refresh(mem)
            bt=to_device(batch,device);opt.zero_grad(set_to_none=True)
            m.train()
            out=m({k:bt[k] for k in ('x_obs','obs_mask','tod','dow','window_start')},mem)
            result=losses(m,bt,out,mem,'full',meta['mean'],meta['std'],cfg)
            result['total'].backward()
            for name,param in m.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(
                        f'FARE_GRADIENT_NONFINITE target={target} step={step+1} parameter={name}'
                    )
            norms={}
            for name,prefix in GROUPS.items():
                values=[float(p.grad.detach().norm().cpu()) for n,p in m.named_parameters()
                        if n.startswith(prefix) and p.grad is not None]
                norms[name]=sum(values) if values else 0.0
            if not all(torch.isfinite(x).all() for x in (result['total'],out['prediction'])):
                raise FloatingPointError('FARE nonfinite gradient diagnostic')
            opt.step()
            processed+=1
            print('FARE_GRADIENT_AUDIT '+json.dumps({'dataset':args.dataset,'target':target,
                 'step':step+1,'retrieval_valid_fraction':float(out['retrieval_valid'].float().mean()),
                 'loss':float(result['total'].detach()),'gradient_norms':norms}),flush=True)
        if processed < args.steps:
            raise RuntimeError(
                f'FARE_GRADIENT_AUDIT_INCOMPLETE target={target}: '
                f'requested {args.steps} steps but only {processed} batches available'
            )
        print(f'FARE_GRADIENT_AUDIT_PASS dataset={args.dataset} target={target} steps={processed}',flush=True)
if __name__=='__main__':main()
