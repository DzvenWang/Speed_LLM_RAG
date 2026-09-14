#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.runners.common import make_run_dir, masked_l1, require_gpu, save_val_outputs, scale_stats, set_seed, validate_stage, as_unicode_array
from baseline.evaluation import evaluate_prediction_file
from baseline.utils import write_json
from baseline.compat.leaf_graph_ops import install_leaf_graph_compat

class LeafDataset(Dataset):
    def __init__(self,cache,split):
        self.cache=Path(cache); self.split=split
        self.x=np.load(self.cache/f'{split}_x_obs_norm.npy',mmap_mode='r')
        self.y=np.load(self.cache/f'{split}_y_norm.npy',mmap_mode='r')
        self.valid=np.load(self.cache/f'{split}_future_valid.npy',mmap_mode='r')
        self.tod=np.load(self.cache/f'{split}_tod_hist.npy',mmap_mode='r')
        self.dow=np.load(self.cache/f'{split}_dow_hist.npy',mmap_mode='r')
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        x=np.asarray(self.x[i],np.float32); H,N=x.shape
        tod=np.asarray(self.tod[i],np.int64)[:,None].repeat(N,axis=1)
        dow=np.asarray(self.dow[i],np.int64)[:,None].repeat(N,axis=1)
        return {
          'x':torch.from_numpy(x[...,None].copy()),'y':torch.from_numpy(np.asarray(self.y[i],np.float32).copy()),
          'valid':torch.from_numpy(np.asarray(self.valid[i],np.uint8).copy()),
          'time_of_day':torch.from_numpy(tod.copy()),'day_of_week':torch.from_numpy(dow.copy()),'idx':torch.tensor(i,dtype=torch.long)
        }

def import_leaf_models(device,adj,cfg):
    vendor=ROOT/'vendor/LEAF'
    if str(vendor) not in sys.path: sys.path.insert(0,str(vendor))
    # cfg.py parses argv at import time; hide Baseline runner arguments from it.
    saved=list(sys.argv); sys.argv=['leaf_external_bridge','--dataset','PEMS03']
    try:
        import cfg as leaf_cfg
        a=leaf_cfg.args
        rc=cfg['runners']['LEAF']; N=int(adj.shape[0])
        a.device=device; a.num_nodes=N; a.in_dim=1; a.out_dim=1
        a.seq_in_len=int(cfg['experiment']['history']); a.seq_out_len=int(cfg['experiment']['horizon']); a.adj=np.asarray(adj,np.float32)
        a.hidden_dim=int(rc['hidden_dim']); a.dropout=float(rc['dropout']); a.stgcn_num_layers=int(rc['stgcn_num_layers'])
        a.hgnn_num_backbone_layers=int(rc['hgnn_num_backbone_layers']); a.hgnn_num_head_layers=int(rc['hgnn_num_head_layers']); a.hgnn_num_hyper_edge=int(rc['hgnn_num_hyper_edge'])
        import model.graph as leaf_graph
        compat=install_leaf_graph_compat(leaf_graph,a,adj)
        print(f'LEAF GRAPH COMPAT={compat}',flush=True)
        from model.graph import GraphBranch
        from model.hypergraph import HypergraphBranch
    finally:
        sys.argv=saved
    return GraphBranch,HypergraphBranch

def move(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) and k!='idx' else v) for k,v in batch.items()}

def run_epoch(model,loader,device,opt=None,clip=5.0):
    train=opt is not None; model.train(train); total=den=0; ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            b=move(b,device); y=b['y'].float(); valid=b['valid'].bool()
            if train: opt.zero_grad(set_to_none=True)
            pred=model(b).squeeze(-1)
            if pred.shape!=y.shape: raise RuntimeError(f'LEAF output {tuple(pred.shape)} != y {tuple(y.shape)}')
            loss=masked_l1(pred,y,valid)
            if train: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),clip); opt.step()
            n=int(valid.sum()); total+=float(loss.detach())*max(n,1); den+=max(n,1)
    return total/max(den,1)

@torch.no_grad()
def predict(model,loader,device,mean,std):
    model.eval(); out=[]
    for b in loader:
        b=move(b,device); p=model(b).squeeze(-1).detach().cpu().numpy(); out.append(p)
    return np.concatenate(out).astype(np.float32)*np.float32(std)+np.float32(mean)

def train_branch(name,Cls,tr,va,device,rc,mean,std):
    model=Cls().to(device); opt=torch.optim.Adam(model.parameters(),lr=float(rc['learning_rate']),weight_decay=float(rc['weight_decay']))
    best=float('inf'); state=None; be=0; stale=0
    for ep in range(1,int(rc['max_epochs'])+1):
        tl=run_epoch(model,tr,device,opt,float(rc['gradient_clip'])); vl=run_epoch(model,va,device,None)
        print(f'LEAF {name} epoch={ep:03d} train_MAE_norm={tl:.6f} val_MAE_norm={vl:.6f}',flush=True)
        if vl<best-1e-8: best=vl; be=ep; stale=0; state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(rc['patience']): print(f'LEAF {name} EARLY STOP epoch={ep} best_epoch={be}',flush=True); break
    if state is None: raise RuntimeError(f'LEAF {name} produced no checkpoint')
    model.load_state_dict(state); model.to(device); raw=predict(model,va,device,mean,std)
    return model,state,be,best,raw

def latest_branch_run(cfg,stage):
    p=Path(cfg['project']['exp_root'])/'LEAF'/cfg['experiment']['subgraph']/stage/'LATEST_BRANCH_RUN.txt'
    if not p.exists(): raise RuntimeError(f'LEAF branch run not found: {p}')
    r=Path(p.read_text().strip())
    if not r.exists(): raise RuntimeError(f'Stale LEAF branch run pointer: {r}')
    return r

def selector_apply(branch_dir,selector_json):
    g=np.load(branch_dir/'graph_val_raw.npy',mmap_mode='r'); h=np.load(branch_dir/'hyper_val_raw.npy',mmap_mode='r')
    data=json.loads(Path(selector_json).read_text()); B,P,N=g.shape
    if len(data)!=B*N: raise RuntimeError(f'LEAF selector JSON entries={len(data)} expected={B*N}')
    out=np.empty((B,P,N),np.float32)
    c=0
    for q in range(B):
        for n in range(N):
            ans=data[c].get('final_answer'); c+=1
            pick=0 if ans is None else int(ans)-1
            if pick not in (0,1): raise RuntimeError(f'Invalid LEAF final_answer={ans}; expected 1/2/None')
            out[q,:,n]=g[q,:,n] if pick==0 else h[q,:,n]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal'],default='development'); ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--phase',choices=['branches','selector'],default='branches'); ap.add_argument('--selector-json',default=None); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    cfg,cache=validate_stage(a.stage,'LEAF'); set_seed(a.seed); device=require_gpu(); rc=cfg['runners']['LEAF']
    if a.phase=='selector':
        if not a.selector_json: raise SystemExit('--selector-json is required for LEAF selector phase')
        br=latest_branch_run(cfg,a.stage); pred=selector_apply(br,a.selector_json); rd=make_run_dir(cfg,'LEAF',a.stage,a.seed)
        sm=save_val_outputs(cfg,cache,rd,pred,'LEAF',a.seed,a.stage,{'phase':'llm_selector_json','branch_run':str(br),'selector_json':str(Path(a.selector_json).resolve()),'selector_contract':'official LEAF-style final_answer 1/2 per query-node','test_access':False})
        print('LEAF SELECTOR COMPLETE',json.dumps({'run_dir':str(rd),'val':sm['val']},default=str),flush=True); return
    trds=LeafDataset(cache,'train'); vads=LeafDataset(cache,'val'); bs=2 if a.smoke else int(rc['batch_size'])
    tr=DataLoader(trds,batch_size=bs,shuffle=True,num_workers=0,pin_memory=True); va=DataLoader(vads,batch_size=bs,shuffle=False,num_workers=0,pin_memory=True)
    adj=np.load(cache/'adjacency.npy'); G,H=import_leaf_models(device,adj,cfg); mean,std=scale_stats(cache)
    rd=make_run_dir(cfg,'LEAF-branches',a.stage,a.seed); t0=time.time()
    if a.smoke:
        for name,Cls in [('GraphBranch',G),('HypergraphBranch',H)]:
            m=Cls().to(device); opt=torch.optim.Adam(m.parameters(),lr=float(rc['learning_rate'])); b=next(iter(tr)); b=move(b,device); p=m(b).squeeze(-1); loss=masked_l1(p,b['y'].float(),b['valid'].bool()); opt.zero_grad(); loss.backward(); opt.step(); print(f'LEAF {name} TFP SMOKE PASS loss={loss.item():.6f} output={tuple(p.shape)}',flush=True)
        return
    results={}
    for name,Cls,key in [('GraphBranch',G,'graph'),('HypergraphBranch',H,'hyper')]:
        m,state,be,best,raw=train_branch(name,Cls,tr,va,device,rc,mean,std); torch.save({'state_dict':state,'best_epoch':be,'best_val_MAE_norm':best},rd/f'{key}_best.pt'); np.save(rd/f'{key}_val_raw.npy',raw.astype(np.float32))
        # canonical branch-only metrics without registering branch as final LEAF result
        from experiment_common.predictions import save_prediction_npz
        meta=pd.read_csv(cache/'val_manifest.csv'); pp=rd/f'{key}_val_predictions.npz'; save_prediction_npz(pp,as_unicode_array(meta.view_id.astype(str).tolist()),raw,manifest_path=rd/f'{key}_val_predictions.manifest.json',metadata={'producer':'Baseline','method':f'LEAF-{name}','split':'val'})
        mt=evaluate_prediction_file(cache,'val',pp); mt.to_csv(rd/f'{key}_val_metrics.csv',index=False); results[key]=mt.loc[mt.family.astype(str)=='ALL'].iloc[0].to_dict()
    write_json(rd/'summary.json',{'artifact_type':'leaf_branch_payload','stage':a.stage,'seed':a.seed,'graph':results['graph'],'hyper':results['hyper'],'official_classes':['vendor/LEAF/model/graph.py::GraphBranch','vendor/LEAF/model/hypergraph.py::HypergraphBranch'],'python':cfg['project']['python'],'isolated_environment':False,'leaf_graph_compat':'baseline/compat/leaf_graph_ops.py::leaf_graph_ops_r1','selector_status':'awaiting_external_json','test_access':False,'wall_seconds':time.time()-t0})
    ptr=Path(cfg['project']['exp_root'])/'LEAF'/cfg['experiment']['subgraph']/a.stage/'LATEST_BRANCH_RUN.txt'; ptr.parent.mkdir(parents=True,exist_ok=True); ptr.write_text(str(rd)+'\n')
    print('LEAF BRANCHES COMPLETE',json.dumps({'run_dir':str(rd),'graph':results['graph'],'hyper':results['hyper']},default=str),flush=True)
if __name__=='__main__': main()
