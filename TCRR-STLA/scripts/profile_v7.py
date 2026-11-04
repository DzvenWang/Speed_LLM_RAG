#!/usr/bin/env python3
"""Synthetic C1/C2 training memory check (never reads real data)."""
import argparse,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tcrr_stla.v7.model import SupportAwareTCRR

def graph(n,k):
    r={}
    for tag in ('upstream','downstream'):
        r[tag+'_ids']=torch.stack([(torch.arange(n)+i+1)%n for i in range(k)],1)
        r[tag+'_valid']=torch.ones(n,k,dtype=torch.bool)
        r[tag+'_gps_log']=torch.ones(n,k)
        r[tag+'_road_log']=torch.ones(n,k)
        r[tag+'_road_valid']=torch.ones(n,k,dtype=torch.bool)
    return r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--nodes',type=int,default=388)
    ap.add_argument('--candidates',type=int,default=12)
    ap.add_argument('--latent-dim',type=int,default=64)
    ap.add_argument('--steps',type=int,default=3)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=ap.parse_args()
    device=torch.device(a.device)
    B,N,K=a.batch_size,a.nodes,a.candidates
    torch.manual_seed(10)
    neigh=torch.stack([(torch.arange(N)+i+1)%N for i in range(8)],1)
    net=SupportAwareTCRR(a.latent_dim,12,12,neigh,graph(N,4),hidden=64,top_k=K).to(device)
    net.freeze_stage('joint_c12')
    args=dict(base=torch.randn(B,12,N,device=device),latent=torch.randn(B,N,a.latent_dim,device=device),
      x=torch.randn(B,12,N,device=device),mask=(torch.rand(B,12,N,device=device)>.4).float(),
      normal=torch.randn(B,K,12,N,device=device)*.1,cf=torch.randn(B,K,12,N,device=device)*.1,
      candidate_scores=torch.randn(B,K,device=device),candidate_coverage=torch.rand(B,K,device=device),
      candidate_valid=torch.ones(B,K,12,N,device=device,dtype=torch.bool),
      candidate_support_raw=torch.rand(B,K,N,14,device=device))
    target=torch.randn(B,12,N,device=device)
    optim=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-3)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    times=[]
    for i in range(a.steps):
        t=time.perf_counter();optim.zero_grad(set_to_none=True)
        out=net.forward_progressive(**args,max_stage=2)
        loss=(out['c12']-target).abs().mean();loss.backward();optim.step()
        if device.type=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-t)
    print(json.dumps({'batch_size':B,'nodes':N,'candidates':K,
        'seconds_per_step':sum(times)/len(times),'queries_per_second':B*len(times)/sum(times),
        'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,
        'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else 0},indent=2))
if __name__=='__main__':main()
