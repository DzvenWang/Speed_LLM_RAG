#!/usr/bin/env python3
"""Real-cache FARE v4 forward/backward benchmark. GPU only, no synthetic claims."""
import _bootstrap,argparse,time,json
from pathlib import Path
import torch
from fare.config import read_config,dump_json,project_path
from fare.pipeline import prepare
from fare.protocol import make_loader,device_batch
from fare.training import forecasting_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='FARE/configs/pilot.yaml')
    ap.add_argument('--target',required=True,choices=['speed','flow'])
    ap.add_argument('--dataset',default='TraffiDent_Net_06')
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--memory',required=True,help='PREBUILT v4 Memory directory')
    ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--out',default='FARE_profile.json')
    a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU profiling must run on compute node with CUDA')
    cfg=read_config(a.config,a.dataset,a.target)
    cfg['experiment']['train_limit']=max(a.steps*cfg['training']['batch_size'],cfg['training']['batch_size'])
    device=torch.device('cuda:0')
    model,memory,train,_,_,_=prepare(cfg,device,project_path(a.checkpoint,must_exist=True),project_path(a.memory,must_exist=True),build=False)
    model.train();optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    loader=make_loader(train,cfg['training']['batch_size'],False,0)
    times=[];peak=0
    for i,raw in enumerate(loader):
        if i>=a.steps:break
        x=device_batch(raw,device);torch.cuda.synchronize();t0=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out=model(x,memory,purge=cfg['memory']['purge_steps'],return_details=False)
        loss,_=forecasting_loss(out,x,cfg)
        loss.backward();optimizer.step()
        torch.cuda.synchronize();times.append(time.perf_counter()-t0)
        peak=max(peak,torch.cuda.max_memory_allocated())
    report={'steps':len(times),'batch_size':cfg['training']['batch_size'],
            'seconds_per_step_avg':sum(times)/max(1,len(times)),
            'peak_allocated_gib':peak/2**30,
            'note':'Synthetic data not used; includes real cache retrieval, transfer and backward'}
    dump_json(project_path(a.out),report);print('FARE4_PROFILE '+json.dumps(report),flush=True)
if __name__=='__main__':main()
