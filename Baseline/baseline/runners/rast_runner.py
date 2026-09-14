#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,sys,time
from collections import deque
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline');sys.path.insert(0,str(ROOT))
from baseline.runners.common import RASTDataset,make_run_dir,masked_l1,require_gpu,save_val_outputs,scale_stats,set_seed,validate_stage
from baseline.backbones.patchstg_bridge import FrozenPatchSTG,PatchSTGRASTBridge
from baseline.compat.rast_optional_backbones import install_rast_native_import_shims

def import_model():
    v=ROOT/'vendor/RAST';sys.path.insert(0,str(v))
    modes=install_rast_native_import_shims()
    print(f'RAST NATIVE IMPORT SHIMS={modes}',flush=True)
    try: from src.arch.rast_arch import RAST
    except Exception as e: raise RuntimeError('RAST import failed after FAISS/native-import compatibility setup. '+repr(e)) from e
    return RAST

def build_model(cfg,variant,seed,device,db):
    c=cfg['runners']['RAST'];N=388;RAST=import_model();m=RAST(num_nodes=N,input_dim=3,output_dim=1,input_len=12,output_len=12,dropout=float(c['dropout']),query_dim=int(c['query_dim']),retrieval_dim=int(c['retrieval_dim']),temporal_dim=int(c['temporal_dim']),spatial_dim=int(c['spatial_dim']),update_interval=int(c['update_interval']),encoder_layers=int(c['encoder_layers']),top_k=int(c['top_k']),database_path=str(db),use_amp=bool(c['use_amp'])).to(device)
    if variant=='patchstg':
        frozen=FrozenPatchSTG(cfg,device);m.backbone=PatchSTGRASTBridge(frozen);m._patchstg_checkpoint=str(frozen.checkpoint);m._patchstg_reference_seed=frozen.reference_seed
    return m

def snapshot_store(model):
    r=model.retrieval_store
    return {'tv':np.asarray(list(r.temporal_vectors),np.float32),'sv':np.asarray(list(r.spatial_vectors),np.float32),'tt':[np.asarray(x,np.float32) for x in getattr(r,'temporal_tensors',[])],'st':[np.asarray(x,np.float32) for x in getattr(r,'spatial_tensors',[])]}

def restore_store(model,snap):
    r=model.retrieval_store;r.temporal_vectors=deque([x for x in snap['tv']],maxlen=r.expected_size);r.spatial_vectors=deque([x for x in snap['sv']],maxlen=r.expected_size);r.temporal_tensors=list(snap.get('tt',[]));r.spatial_tensors=list(snap.get('st',[]));r._rebuild_indices()

def save_store(path,snap):
    kw={'temporal_vectors':snap['tv'],'spatial_vectors':snap['sv'],'nt':np.int64(len(snap['tt'])),'ns':np.int64(len(snap['st']))}
    for i,x in enumerate(snap['tt']):kw[f'tt_{i}']=x
    for i,x in enumerate(snap['st']):kw[f'st_{i}']=x
    np.savez_compressed(path,**kw)

def load_store(path):
    with np.load(path,allow_pickle=False) as z:
        nt=int(z['nt']);ns=int(z['ns']);return {'tv':z['temporal_vectors'],'sv':z['spatial_vectors'],'tt':[z[f'tt_{i}'] for i in range(nt)],'st':[z[f'st_{i}'] for i in range(ns)]}

def run_epoch(model,dl,device,ep,train,opt=None):
    model.train(train);tot=den=0;ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx,x,y,valid in dl:
            x=x.float().to(device);y=y.float().to(device);valid=valid.bool().to(device)
            if train:opt.zero_grad(set_to_none=True)
            out=model(x,None,int(idx[0]) if len(idx) else 0,ep,train);p=out['prediction'].squeeze(-1)
            loss=masked_l1(p,y,valid)
            if train:loss.backward();nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],5.0);opt.step()
            n=int(valid.sum());tot+=float(loss.item())*max(n,1);den+=max(n,1)
    return tot/max(den,1)
def predict(model,dl,device,mean,std,ep):
    out=[];model.eval()
    with torch.no_grad():
        for idx,x,_y,_valid in dl: out.append(model(x.float().to(device),None,int(idx[0]) if len(idx) else 0,ep,False)['prediction'].squeeze(-1).cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)*np.float32(std)+np.float32(mean)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','formal'],default='formal');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--variant',choices=['native','patchstg'],default='native');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.variant=='patchstg' and a.stage!='formal':raise SystemExit('RAST-PatchSTG controlled adaptation is formal-only')
    cfg,cache=validate_stage(a.stage,'RAST');set_seed(a.seed);device=require_gpu();c=cfg['runners']['RAST'];label='RAST' if a.variant=='native' else 'RAST-PatchSTG'
    tr=RASTDataset(cache,'train');va=RASTDataset(cache,'val');dl=lambda d,s:DataLoader(d,batch_size=int(c['batch_size']),shuffle=s,num_workers=0,pin_memory=True);trl,val=dl(tr,True),dl(va,False)
    temp=Path(cfg['project']['exp_root'])/'_tmp_rast'/f'{label}_{a.seed}_{int(time.time())}';temp.mkdir(parents=True,exist_ok=True);m=build_model(cfg,a.variant,a.seed,device,temp);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    if a.smoke:print('RAST GPU SMOKE',run_epoch(m,trl,device,1,True,opt));return
    best=float('inf');state=store=None;be=0;stale=0;t0=time.time()
    for ep in range(1,int(c['max_epochs'])+1):
        trlss=run_epoch(m,trl,device,ep,True,opt);vl=run_epoch(m,val,device,ep,False,None);print(f'EPOCH {ep:03d} train_MAE_norm={trlss:.6f} val_MAE_norm={vl:.6f}',flush=True)
        if vl<best-1e-8:best=vl;be=ep;stale=0;state=copy.deepcopy({k:v.detach().cpu() for k,v in m.state_dict().items()});store=snapshot_store(m)
        else:
            stale+=1
            if stale>=int(c['patience']):print(f'EARLY STOP epoch={ep} best_epoch={be}');break
    m.load_state_dict(state);restore_store(m,store);mean,std=scale_stats(cache);raw=predict(m,val,device,mean,std,be);rd=make_run_dir(cfg,label,a.stage,a.seed);torch.save({'model':state,'best_epoch':be,'best_val_mae_norm':best,'variant':a.variant},rd/'best.pt');save_store(rd/'retrieval_store_best.npz',store)
    method_key='RAST' if a.variant=='native' else 'RAST-PatchSTG'
    # save_val_outputs expects cfg method entry; v0.5 config contains both
    sm=save_val_outputs(cfg,cache,rd,raw,method_key,a.seed,a.stage,{'best_epoch':be,'best_val_MAE_norm':best,'params':sum(p.numel() for p in params),'wall_seconds':time.time()-t0,'official_architecture':'vendor/RAST/src/arch/rast_arch.py::RAST','variant':a.variant,'retrieval_memory':'training batches only','checkpoint_selection':'val_only','test_access':False,'patchstg_reference_seed':getattr(m,'_patchstg_reference_seed',None),'patchstg_checkpoint':getattr(m,'_patchstg_checkpoint',None)})
    print('RAST COMPLETE',json.dumps({'run_dir':str(rd),'val':sm['val']},default=str))
if __name__=='__main__':main()
