#!/usr/bin/env python3
"""Pure CPU/CUDA C1 contract test at the actual node count; no data/teacher cache reads.

Pass --device cuda after GPU allocation for a real AMP optimizer step; CPU is the
mandatory prequeue test. This is a protocol test, not a real-traffic MAE claim.
"""
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tcrr_stla.v7.model import SupportAwareTCRR
from tcrr_stla.v7.losses import cfrm_auxiliary,full_activation_targets


def synthetic_road(n,k):
    a={}
    for name,shift in [('upstream',1),('downstream',-1)]:
        ids=torch.arange(n).unsqueeze(1).expand(n,k).clone()
        ids[:,0]=(torch.arange(n)+shift)%n
        valid=torch.zeros(n,k,dtype=torch.bool);valid[:,0]=True
        a[name+'_ids']=ids;a[name+'_valid']=valid
        a[name+'_gps_log']=torch.zeros(n,k)
        a[name+'_road_log']=torch.zeros(n,k)
        a[name+'_road_valid']=valid.clone()
    return a


def run(nodes,model_dim,top_k,hidden,device='cpu',batch_size=2,amp=False):
    torch.manual_seed(20261104)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    dev=torch.device(device)
    n=int(nodes);k=int(top_k);B=int(batch_size);H=P=12
    model=SupportAwareTCRR(model_dim,P,H,
        torch.arange(n)[:,None].expand(n,2),synthetic_road(n,2),
        hidden=hidden,top_k=k).to(dev)
    model.freeze_stage('c1')
    # Real-size tensors, including all-invalid nodes and random faults.
    raw=torch.randn(B,k,P,n,device=dev)*.12
    available=(torch.rand(B,k,P,n,device=dev)>.15)
    available[:,:,:,0]=False
    inp=dict(base=torch.randn(B,P,n,device=dev),
             latent=torch.randn(B,n,model_dim,device=dev),
             x=torch.randn(B,H,n,device=dev),
             mask=(torch.rand(B,H,n,device=dev)>.4).float(),
             normal=raw,cf=torch.randn_like(raw)*.04,
             candidate_scores=torch.randn(B,k,device=dev),
             candidate_coverage=torch.rand(B,k,device=dev),
             candidate_valid=available,
             candidate_support_raw=torch.randn(B,k,n,14,device=dev))
    target=torch.randn(B,P,n,device=dev)
    valid=torch.ones(B,P,n,device=dev,dtype=torch.bool)
    valid[:,-1,0]=False
    target[:,-1,0]=float('nan')
    batch={'y_norm':target,'future_valid':valid}
    optim=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    scaler=torch.amp.GradScaler('cuda',enabled=(device=='cuda' and amp))
    for step in range(2):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=scaler.is_enabled()):
            out=model.forward_progressive(**inp,max_stage=1)
            clean_y=torch.where(valid,target,torch.zeros_like(target))
            diff=torch.where(valid,out['c1']-clean_y,torch.zeros_like(clean_y))
            forecast=diff.abs().sum()/valid.sum().clamp_min(1)
            aux=cfrm_auxiliary(out,batch,0.,1.,target_temperature=.25)
            loss=forecast+aux
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite C1 loss')
        if scaler.is_enabled():
            scaler.scale(loss).backward();scaler.unscale_(optim)
        else:loss.backward()
        for key in ('context_proj.0.weight','c1_activation','transport.marginal_head.0.weight'):
            p=dict(model.named_parameters())[key]
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError('Gradient missing/nonfinite: '+key)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),5.)
        if scaler.is_enabled():scaler.step(optim);scaler.update()
        else:optim.step()
        if not torch.isfinite(out['candidate_weights']).all():raise RuntimeError('Nonfinite weights')
        if not torch.allclose(out['candidate_weights'][:,-1,:,0],
                              torch.ones_like(out['candidate_weights'][:,-1,:,0]),atol=2e-5):
            raise RuntimeError('All-invalid candidate node not routed to None')
    info={'device':device,'nodes':n,'k':k,'batch':B,'amp':amp,
          'peak_gpu_gib':round(torch.cuda.max_memory_allocated()/1024**3,3) if dev.type=='cuda' else None}
    print('V71_C1_PREFLIGHT_PASS '+json.dumps(info),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--nodes',type=int,default=388)
    p.add_argument('--model-dim',type=int,default=64)
    p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--amp',action='store_true')
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested, unavailable')
    run(a.nodes,a.model_dim,a.top_k,a.hidden,a.device,a.batch_size,a.amp)
if __name__=='__main__':main()
