#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline');sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
from baseline.runners.common import GinARDataset,ImputationDataset,RASTDataset,SARAFTensorDataset,build_time_marks,save_split_outputs,scale_stats,set_seed,require_gpu,simple_namespace
from baseline.runners.ginar_runner import import_model as ginar_import,official_doubletransition
from baseline.runners.imputeformer_runner import build_model as impute_build,forecast_val as impute_forecast
from baseline.runners.raft_runner import import_model as raft_import,make_protocol_retrieve
from baseline.runners.saraf_runner import import_model as saraf_import,overlap_masker,_probe_time_bonus
from baseline.runners.rast_runner import build_model as rast_build,load_store,restore_store,predict as rast_predict
from baseline.backbones.patchstg_bridge import FrozenPatchSTG
from baseline.run_identity import load_cache_identity
from baseline.shared_cache import cache_path
from baseline.backbones.frozen_registry import sha256_file
from types import MethodType

def load_sel(cfg,m,seed):
    exp=Path(cfg['project']['exp_root']);p=exp/'FORMAL_SELECTION.json'
    o=json.loads(p.read_text())
    expected=load_cache_identity(cache_path(cfg,'formal'),cfg['version'])
    if o.get('formal_run_identity')!=expected:
        raise RuntimeError('FORMAL_SELECTION.json does not belong to the current v0.6.0 formal cache identity')
    x=o['methods'][m][str(seed)];cp=Path(x['checkpoint'])
    if not cp.exists():raise FileNotFoundError(cp)
    got=sha256_file(cp)
    if got!=str(x.get('checkpoint_sha256','')):
        raise RuntimeError(f'Formal checkpoint SHA256 mismatch for {m} seed={seed}')
    return x,torch.load(cp,map_location='cpu',weights_only=False)
def ginar(cfg,fc,ec,seed,device,ck):
    ds=GinARDataset(ec,'test');dl=DataLoader(ds,batch_size=int(cfg['runners']['GinAR']['batch_size']),shuffle=False);adj=np.load(fc/'adjacency.npy');M=ginar_import();c=cfg['runners']['GinAR'];m=M(12,adj.shape[0],12,3,int(c['emb_size']),int(c['graph_size']),int(c['layers']),float(c['dropout']),official_doubletransition(adj,device)).to(device);m.load_state_dict(ck['model']);m.eval();mean,std=scale_stats(fc);out=[]
    with torch.no_grad():
        for feat,_y,_v,_i in dl:out.append(m(feat.float().to(device)).cpu().numpy())
    return np.concatenate(out)*np.float32(std)+np.float32(mean)
def impute(cfg,fc,ec,seed,device,ck):
    ds=ImputationDataset(ec,'test');dl=DataLoader(ds,batch_size=int(cfg['runners']['ImputeFormer']['batch_size']),shuffle=False);m=impute_build(cfg,ds.x.shape[2]).to(device);m.load_state_dict(ck['model']);patch=FrozenPatchSTG(cfg,device);return impute_forecast(m,dl,patch,device)
def raft(cfg,fc,ec,seed,device,ck):
    txm,tym,_=build_time_marks(cfg,fc,'train');qxm,qym,_=build_time_marks(cfg,ec,'test');mem=SARAFTensorDataset(fc,'train',txm,tym,'full');q=SARAFTensorDataset(ec,'test',qxm,qym,'obs');c=cfg['runners']['RAFT'];Model=raft_import();mc=simple_namespace(task_name='long_term_forecast',seq_len=12,pred_len=12,enc_in=388,n_period=int(c['n_period']),topm=int(c['topm']),gpu=0);m=Model(mc).to(device);rt=m.rt;rt.prepare_dataset(mem);span=24;rt.retrieve=MethodType(make_protocol_retrieve(mem.meta.window_start.astype(int).to_numpy(),mem.meta.window_start.astype(int).to_numpy(),q.meta.window_start.astype(int).to_numpy(),span),rt);va=rt.retrieve_all(q,train=False,device=device);m.retrieval_dict={'valid':va.to(device)};del m.rt;m.load_state_dict(ck['model']);dl=DataLoader(q,batch_size=int(c['batch_size']),shuffle=False);mean,std=scale_stats(fc);out=[];m.eval()
    with torch.no_grad():
        for idx,x,_y,_xm,_ym in dl:out.append(m(x.float().to(device),idx.long().to(device),mode='valid').cpu().numpy())
    return np.concatenate(out)*np.float32(std)+np.float32(mean)
def saraf(cfg,fc,ec,seed,device,ck):
    txm,tym,_=build_time_marks(cfg,fc,'train');qxm,qym,_=build_time_marks(cfg,ec,'test');mem=SARAFTensorDataset(fc,'train',txm,tym,'full');q=SARAFTensorDataset(ec,'test',qxm,qym,'obs');c=cfg['runners']['SARAF'];Model=saraf_import();mc=simple_namespace(task_name='long_term_forecast',seq_len=12,pred_len=12,features=str(c.get('features','M')),enc_in=388,n_period=int(c.get('n_period',1)),topm=int(c.get('topm',20)),freq=str(c.get('freq','5min')),time_aware_weight=float(c.get('time_aware_weight',0.3)),fusion_method=str(c.get('fusion_method','weighted')),use_gpu=True,gpu=0);m=Model(mc).to(device);rt=m.rt;rt.steps_per_day=288;rt.steps_per_year=288*365;rt.prepare_dataset(mem);_probe_time_bonus(rt,q,device);span=24;rt.masked_sim=MethodType(overlap_masker(q.meta.window_start.astype(int).to_numpy(),mem.meta.window_start.astype(int).to_numpy(),span),rt);va=rt.retrieve_all(q,train=True,device=device);m.retrieval_temporal_dict={'valid':va.to(device)};del m.rt;m.load_state_dict(ck['model']);dl=DataLoader(q,batch_size=int(c['batch_size']),shuffle=False);mean,std=scale_stats(fc);out=[];m.eval()
    with torch.no_grad():
        for idx,x,_y,xm,_ym in dl:out.append(m(x.float().to(device),xm.float().to(device),idx.long().to(device),mode='valid').cpu().numpy())
    return np.concatenate(out)*np.float32(std)+np.float32(mean)
def rast(cfg,fc,ec,seed,device,ck,sel):
    ds=RASTDataset(ec,'test');dl=DataLoader(ds,batch_size=int(cfg['runners']['RAST']['batch_size']),shuffle=False);rd=Path(sel['run_dir']);m=rast_build(cfg,'native',seed,device,rd/'evaluation_database');m.load_state_dict(ck['model']);restore_store(m,load_store(rd/'retrieval_store_best.npz'));mean,std=scale_stats(fc);return rast_predict(m,dl,device,mean,std,int(ck['best_epoch']))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['ImputeFormer','GinAR','RAFT','RAST','SARAF'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();cfg=load_yaml(ROOT/'configs/baseline.yaml');set_seed(a.seed);device=require_gpu();exp=Path(cfg['project']['exp_root']);fc=cache_path(cfg,'formal');ec=cache_path(cfg,'evaluation');sel,ck=load_sel(cfg,a.method,a.seed);fn={'GinAR':ginar,'ImputeFormer':impute,'RAFT':raft,'SARAF':saraf};raw=rast(cfg,fc,ec,a.seed,device,ck,sel) if a.method=='RAST' else fn[a.method](cfg,fc,ec,a.seed,device,ck);rd=exp/'evaluation'/a.method/cfg['experiment']['subgraph']/f'seed{a.seed}';rd.mkdir(parents=True,exist_ok=True);sm=save_split_outputs(cfg,ec,rd,'test',raw,a.method,a.seed,{'formal_checkpoint':sel['checkpoint'],'formal_checkpoint_sha256':sel['checkpoint_sha256'],'test_access':'evaluation_only_after_freeze'});print(json.dumps(sm,default=str,indent=2))
if __name__=='__main__':main()
