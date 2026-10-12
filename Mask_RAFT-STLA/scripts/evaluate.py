#!/usr/bin/env python3
"""Evaluate the exact saved v2 config; only Train Memory is consulted."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import cache_path,relocate_saved_config,file_sha256
from mask_raft_stla.data import SupportSplit,RetrievalCache
from mask_raft_stla.evidence import FutureMemory
from mask_raft_stla.normal_teacher import prepare_normal_teacher_bank
from mask_raft_stla.trainer import build_model,_make_loader,predict
from mask_raft_stla.evaluation import summarize_by_family,save_summary
from scripts.build_retrieval_cache import ensure_cache


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--target',choices=['speed','flow'])
    ap.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    ap.add_argument('--backbone-checkpoint')
    ap.add_argument('--evidence-mode',choices=['full','no_evidence','shuffled','random_future','cross_history','no_risk','standard_future','no_mask_replay','no_rerank','no_retrieval'])
    a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if str(ck.get('version',''))!='2.3.0':
        raise RuntimeError('Only v2 checkpoints can be evaluated by v2 scripts')
    cfg=relocate_saved_config(ck['config'])
    if a.target and a.target!=cfg['experiment']['target']:raise RuntimeError('Target must match saved checkpoint')
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_path=Path(a.backbone_checkpoint or os.environ.get('BACKBONE_CHECKPOINT') or ck['backbone_checkpoint'])
    if not backbone_path.is_file():raise FileNotFoundError(f'STLA checkpoint unavailable: {backbone_path}; pass --backbone-checkpoint')
    if ck.get('backbone_sha256') and file_sha256(backbone_path)!=ck['backbone_sha256']:
        raise RuntimeError('Incorrect STLA Backbone checkpoint: SHA256 differs from training')
    model,_=build_model(cfg,backbone_path,device)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    # Train used an immutable reference teacher for cacheable mask-replay. Do
    # not accidentally use the subsequently fine-tuned query backbone here.
    teacher=None
    if cfg['runtime'].get('frozen_replay_teacher',False) and (a.evidence_mode or cfg['training'].get('ablation','full'))!='no_retrieval':
        initial,_=build_model(cfg,backbone_path,device)
        teacher=initial.stla.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        print('EVAL_REPLAY_TEACHER fixed_pretrained_backbone',flush=True)
    mean=float(ck['mean']);std=float(ck['std'])
    formal=cache_path(cfg,'formal');store=FutureMemory(formal)
    if cfg['runtime'].get('preload_retrieval_memory_gpu',False):
        print(f'EVAL_RETRIEVAL_MEMORY_GPU loaded={store.preload(device)}',flush=True)
    mode=a.evidence_mode or cfg['training'].get('ablation','full')
    normal_bank=prepare_normal_teacher_bank(cfg,store,teacher,backbone_path,device) if teacher is not None and mode!='no_retrieval' else None
    os.environ['BACKBONE_CHECKPOINT']=str(backbone_path)
    rows=[]
    for sev in a.severities:
        c=cache_path(cfg,'evaluation',sev)
        r=None if mode=='no_retrieval' else ensure_cache(cfg,c,'test',severity=sev)
        ds=SupportSplit(c,'test');dl=_make_loader(ds,cfg['training']['batch_size'],False,cfg)
        p=predict(model,dl,RetrievalCache(r) if r else None,store,device,mean,std,cfg,mode,teacher=teacher,normal_bank=normal_bank)
        fam=summarize_by_family(p['prediction'],p['true'],p['valid'],p['direct'],ds.meta)
        base=summarize_by_family(p['base'],p['true'],p['valid'],p['direct'],ds.meta)
        out=Path(a.checkpoint).resolve().parent/'evaluation'/f's{sev}'/mode;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'test_predictions.npz',prediction=p['prediction'].astype(np.float32),
             base_prediction=p['base'].astype(np.float32),correction=p['correction'].astype(np.float32))
        save_summary(out/'summary.json',{'severity':sev,'mode':mode,'checkpoint_sha256':file_sha256(a.checkpoint),'metrics':fam,'backbone':base,'gate_mean':p['gate_mean'],'useful_candidate_recall':p['useful_candidate_recall'],'weighted_candidate_utility_normalized':p['weighted_candidate_utility_normalized']})
        rows.append({'severity':sev,'mode':mode,**fam['ALL'],**{'base_'+k:v for k,v in base['ALL'].items()}})
        print('EVAL '+json.dumps(rows[-1]),flush=True)
    out=Path(a.checkpoint).resolve().parent/'evaluation';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f'severity_summary.{mode}.csv',index=False)
    if mode==cfg['training'].get('ablation','full'):
        pd.DataFrame(rows).to_csv(out/'severity_summary.csv',index=False)
        root=Path(a.checkpoint).resolve().parent/'summary.json'
        payload=json.loads(root.read_text()) if root.is_file() else {}
        payload.update(test_evaluated=True,evaluation_summary=rows)
        save_summary(root,payload)
    print(f'EVALUATION_PASS mode={mode} summary={out / f"severity_summary.{mode}.csv"}',flush=True)
if __name__=='__main__':main()
