#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
sys.path.insert(0,str(ROOT))
from baseline.runners.common import (  # noqa: E402
    SARAFTensorDataset, build_time_marks, make_run_dir, masked_mse, require_gpu,
    save_val_outputs, scale_stats, set_seed, simple_namespace, validate_stage,
)


def import_model():
    vendor=ROOT/'vendor'/'RAFT'; sys.path.insert(0,str(vendor))
    try:
        from models.RAFT import Model
    except Exception as e:
        raise RuntimeError(f'RAFT official import failed from {vendor}: {e!r}') from e
    return Model


def make_protocol_retrieve(memory_starts: np.ndarray, train_query_starts: np.ndarray, val_query_starts: np.ndarray, span: int):
    """Official RAFT retrieval logic with only its train leakage mask replaced.

    Stock RAFT masks neighboring *row indices*, which is valid only for contiguous
    sliding-window datasets. Shared Protocol rows are balanced/reordered, so we mask
    true time-interval overlap using frozen ``window_start`` instead.
    """
    mem=torch.as_tensor(memory_starts,dtype=torch.long)
    qry_train=torch.as_tensor(train_query_starts,dtype=torch.long)
    qry_val=torch.as_tensor(val_query_starts,dtype=torch.long)

    def retrieve(self,x,index,train=True):
        index=index.to(x.device); bsz,seq_len,channels=x.shape
        if seq_len!=self.seq_len or channels!=self.channels:
            raise RuntimeError(f'RAFT input contract failed {(bsz,seq_len,channels)} expected (*,{self.seq_len},{self.channels})')
        x_mg,_=self.decompose_mg(x)
        sim=self.periodic_batch_corr(self.train_data_all_mg.flatten(start_dim=2),x_mg.flatten(start_dim=2))
        qry = qry_train if train else qry_val
        qs=qry[index.detach().cpu()].to(x.device); ms=mem.to(x.device)
        overlap=(ms.unsqueeze(0)<(qs+span).unsqueeze(1)) & (qs.unsqueeze(1)<(ms+span))
        sim=sim.masked_fill(overlap.unsqueeze(0),float('-inf'))
        sim_flat=sim.reshape(self.n_period*bsz,self.n_train)
        k=min(int(self.topm),int(self.n_train))
        topm_index=torch.topk(sim_flat,k,dim=1).indices
        ranking_sim=torch.ones_like(sim_flat)*float('-inf')
        rows=torch.arange(sim_flat.size(0),device=sim_flat.device).unsqueeze(-1)
        ranking_sim[rows,topm_index]=sim_flat[rows,topm_index]
        ranking_sim=ranking_sim.reshape(self.n_period,bsz,self.n_train)
        ranking_prob=F.softmax(ranking_sim/self.temperature,dim=2).detach().cpu()
        y_data_all=self.y_data_all_mg.flatten(start_dim=2)
        pred=torch.bmm(ranking_prob,y_data_all).reshape(self.n_period,bsz,-1,channels)
        return pred.to(x.device)
    return retrieve


def precompute(model,memory_ds,train_ds,val_ds,cfg,device):
    rt=model.rt; rt.prepare_dataset(memory_ds)
    span=int(cfg['experiment']['history'])+int(cfg['experiment']['horizon'])
    rt.retrieve=MethodType(make_protocol_retrieve(memory_ds.meta.window_start.astype(int).to_numpy(),train_ds.meta.window_start.astype(int).to_numpy(),val_ds.meta.window_start.astype(int).to_numpy(),span),rt)
    print('RAFT RETRIEVAL train queries -> Train-only full-history memory',flush=True)
    tr=rt.retrieve_all(train_ds,train=True,device=device)
    print('RAFT RETRIEVAL val queries -> Train-only full-history memory',flush=True)
    va=rt.retrieve_all(val_ds,train=False,device=device)
    model.retrieval_dict={'train':tr.detach().to(device, non_blocking=True),'valid':va.detach().to(device, non_blocking=True)}
    del model.rt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {'train_retrieval_shape':list(tr.shape),'val_retrieval_shape':list(va.shape),'memory_rows':len(memory_ds),'overlap_exclusion_span_steps':span}


def mkloader(ds,batch,shuffle): return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=True,drop_last=False)


def run_epoch(model,dl,device,mode,optimizer=None):
    train=optimizer is not None; model.train(train); total=0.;den=0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx,x,y,_xm,_ym in dl:
            idx=idx.long().to(device); x=x.float().to(device); y=y.float().to(device)
            ii=idx.detach().cpu().numpy(); valid=torch.from_numpy(np.asarray(dl.dataset.valid[ii],np.uint8)).bool().to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            p=model(x,idx,mode=mode)
            if p.shape!=y.shape: raise RuntimeError(f'RAFT output contract failed pred={tuple(p.shape)} y={tuple(y.shape)}')
            loss=masked_mse(p,y,valid)
            if train: loss.backward(); optimizer.step()
            n=int(valid.sum().item()); total+=float(loss.detach().item())*max(n,1); den+=max(n,1)
    return total/max(den,1)


def predict(model,dl,device,mean,std):
    out=[];model.eval()
    with torch.no_grad():
        for idx,x,_y,_xm,_ym in dl:
            p=model(x.float().to(device),idx.long().to(device),mode='valid');out.append(p.detach().cpu().numpy())
    p=np.concatenate(out,axis=0).astype(np.float32);return p*np.float32(std)+np.float32(mean)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development','formal']);ap.add_argument('--seed',type=int,default=None);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg,cache=validate_stage(args.stage,'RAFT');rcfg=cfg['runners']['RAFT'];seed=int(cfg['experiment']['seed'] if args.seed is None else args.seed);set_seed(seed);device=require_gpu()
    txm,tym,_=build_time_marks(cfg,cache,'train');vxm,vym,_=build_time_marks(cfg,cache,'val')
    lm=256 if args.smoke else None;lq=64 if args.smoke else None
    memory=SARAFTensorDataset(cache,'train',txm,tym,'full',limit=lm);train=SARAFTensorDataset(cache,'train',txm,tym,'obs',limit=lq);val=SARAFTensorDataset(cache,'val',vxm,vym,'obs',limit=lq)
    H=int(cfg['experiment']['history']);P=int(cfg['experiment']['horizon']);N=int(train.x.shape[2]);Model=import_model()
    mc=simple_namespace(task_name='long_term_forecast',seq_len=H,pred_len=P,enc_in=N,n_period=int(rcfg['n_period']),topm=int(rcfg['topm']),gpu=0)
    model=Model(mc).to(device);nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'RAFT START stage={args.stage} seed={seed} device={device} params={nparams} H={H} P={P} N={N}',flush=True)
    print('RAFT SAFETY Train-only memory; external Val-only model selection; no Test dataset instantiated.',flush=True)
    ri=precompute(model,memory,train,val,cfg,device);batch=int(rcfg['batch_size']);trl=mkloader(train,batch,True);val=mkloader(val,batch,False);opt=torch.optim.Adam(model.parameters(),lr=float(rcfg['learning_rate']))
    if args.smoke:
        idx,x,y,_xm,_ym=next(iter(trl));idx=idx.long().to(device);x=x.float().to(device);y=y.float().to(device);vv=torch.from_numpy(np.asarray(trl.dataset.valid[idx.cpu().numpy()],np.uint8)).bool().to(device)
        opt.zero_grad(set_to_none=True);p=model(x,idx,mode='train');loss=masked_mse(p,y,vv);loss.backward();opt.step();print(f'RAFT GPU SMOKE PASS loss={loss.item():.6f} output={tuple(p.shape)} retrieval={ri}',flush=True);return
    best=float('inf');state=None;be=0;stale=0;t0=time.time();pat=int(rcfg['patience'])
    for ep in range(1,int(rcfg['max_epochs'])+1):
        et=time.time();tr=run_epoch(model,trl,device,'train',opt);va=run_epoch(model,val,device,'valid',None);print(f'EPOCH {ep:03d} train_MSE_norm={tr:.6f} val_MSE_norm={va:.6f} sec={time.time()-et:.1f}',flush=True)
        if va<best-1e-10: best=va;be=ep;stale=0;state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=pat: print(f'EARLY STOP epoch={ep} best_epoch={be}',flush=True);break
    if state is None: raise RuntimeError('RAFT did not produce checkpoint')
    model.load_state_dict(state);model.to(device);mean,std=scale_stats(cache);raw=predict(model,val,device,mean,std);rd=make_run_dir(cfg,'RAFT',args.stage,seed);torch.save({'model':state,'best_epoch':be,'best_val_mse_norm':best},rd/'best.pt')
    sm=save_val_outputs(cfg,cache,rd,raw,'RAFT',seed,args.stage,{'best_epoch':be,'best_val_MSE_norm':best,'params':nparams,'wall_seconds':time.time()-t0,'official_architecture':'vendor/RAFT/models/RAFT.py::Model','official_retrieval':'vendor/RAFT/layers/Retrieval.py::RetrievalTool','retrieval':ri,'retrieval_memory_split':'train','retrieval_memory_signal':'full_train_history','query_signal':'incomplete_history','train_overlap_mask':'Protocol window_start interval exclusion','checkpoint_selection':'val_only','test_access':False})
    print('RAFT COMPLETE',json.dumps({'run_dir':str(rd),'val':sm['val']},default=str),flush=True)
if __name__=='__main__': main()
