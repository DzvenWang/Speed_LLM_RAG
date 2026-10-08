#!/usr/bin/env python3
from pathlib import Path
import argparse, json, os, subprocess, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import load_config, cache_path
from mask_raft_stla.data import SupportSplit, RetrievalCache
from mask_raft_stla.trainer import build_model, _batch_to_device
from mask_raft_stla.evaluation import summarize_by_family, save_summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--target',choices=['speed','flow']); ap.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90]); ap.add_argument('--config',default=None); a=ap.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); base_ck=Path(ck['backbone_checkpoint']); model,_=build_model(cfg,base_ck,device); model.load_state_dict(ck['state_dict'],strict=True); model.to(device).eval(); mean=float(ck['mean']);std=float(ck['std'])
    rows=[]
    for sev in a.severities:
        cache=cache_path(cfg,'evaluation',sev); rpath=Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/f'eval_s{sev}'/'test.mask_raft_topk.npz'
        if not rpath.exists(): subprocess.check_call([sys.executable,'-u',str(ROOT/'scripts/build_retrieval_cache.py'),'--target',cfg['experiment']['target'],'--severity',str(sev)],cwd=ROOT)
        ds=SupportSplit(cache,'test')
        nw=int(cfg.get('runtime',{}).get('num_workers',0)); pin=bool(cfg.get('runtime',{}).get('pin_memory',True)) and torch.cuda.is_available()
        dl_kwargs=dict(batch_size=int(cfg['training']['batch_size']),shuffle=False,num_workers=nw,pin_memory=pin)
        if nw>0:
            dl_kwargs['persistent_workers']=bool(cfg.get('runtime',{}).get('persistent_workers',True))
            dl_kwargs['prefetch_factor']=int(cfg.get('runtime',{}).get('prefetch_factor',2))
        dl=DataLoader(ds,**dl_kwargs); rc=RetrievalCache(rpath); memory_y=np.load(cache_path(cfg,'formal')/'train_y_norm.npy',mmap_mode='r')
        ps=[];ys=[];vs=[];dsx=[];g=[]
        with torch.no_grad():
            for b in dl:
                idx=b['index'].numpy(); ev,conf=rc.batch_evidence(idx,memory_y,float(cfg['retrieval']['temperature'])); b=_batch_to_device(b,device); o=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),ev.to(device).float(),conf.to(device).float()); ps.append((o['prediction']*std+mean).cpu().numpy());ys.append(b['y_raw'].cpu().numpy());vs.append(b['future_valid'].cpu().numpy());dsx.append(b['direct_mask'].cpu().numpy());g.append(float(o['gate'].mean()))
        p=np.concatenate(ps);y=np.concatenate(ys);v=np.concatenate(vs);d=np.concatenate(dsx); fam=summarize_by_family(p,y,v,d,ds.meta); outdir=Path(a.checkpoint).resolve().parent/'evaluation'/f's{sev}';outdir.mkdir(parents=True,exist_ok=True);np.savez_compressed(outdir/'test_predictions.npz',prediction=p.astype(np.float32));save_summary(outdir/'summary.json',{'severity':sev,'metrics':fam,'gate_mean':float(np.mean(g))}); rows.append({'severity':sev,**fam['ALL']}); print(f'EVAL s{sev} '+json.dumps(fam['ALL']),flush=True)
    run_dir=Path(a.checkpoint).resolve().parent; eval_dir=run_dir/'evaluation'; eval_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(eval_dir/'severity_summary.csv',index=False)
    root_summary=run_dir/'summary.json'
    payload=json.loads(root_summary.read_text(encoding='utf-8')) if root_summary.is_file() else {}
    payload.update({'test_evaluated':True,'evaluation_severities':[int(x) for x in a.severities],'evaluation_summary':rows})
    save_summary(root_summary,payload)
    print(f'EVALUATION_PASS checkpoint={a.checkpoint} severities={",".join(str(x) for x in a.severities)} summary={eval_dir/"severity_summary.csv"}',flush=True)
if __name__=='__main__': main()
