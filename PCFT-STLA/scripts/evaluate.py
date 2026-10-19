#!/usr/bin/env python3
"""Evaluate a saved PCFT v1.0 Full checkpoint without retraining.

Safe defaults deliberately override old checkpoint runtime settings, which
were tuned for training and caused GPU OOM during automatic evaluation.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcft_stla.config import cache_path, relocate_saved_config, file_sha256
from pcft_stla.data import SupportSplit, FullTrainRepository
from pcft_stla.trainer import build_model, make_loader, predict
from pcft_stla.memory import LatentMemoryIndex
from pcft_stla.evaluation import summarize_by_family, save_summary
from pcft_stla.runtime_safety import safe_evaluation_config, predict_with_retry

MODES=(
    'full','no_retrieval','no_future','no_regional','global_only',
    'no_none','shuffled','random_future','cross_history','no_innovation',
    'no_risk','legacy_v23','no_prop_retrieval','no_lag_alignment',
    'no_prop_control','no_prop_support','legacy_only',
    'propagation_only','observed_support_only','no_local_regime','no_directional_graph','no_action_utility'
)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--target', choices=['speed','flow'])
    p.add_argument('--severities', nargs='+', type=int, default=[20,40,60,90])
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--evidence-mode', choices=MODES)
    p.add_argument('--batch-size',type=int,default=None,
                   help='Safe default is 8; on GPU OOM automatically retry at 4/2/1')
    p.add_argument('--num-workers',type=int,default=0,
                   help='0 disables multi-process shared-memory DataLoader workers')
    p.add_argument('--preload-repository-gpu',action='store_true',
                   help='Opt-in only; OFF by default to avoid GPU OOM')
    a=p.parse_args()
    if not a.severities or any(s not in (20,40,60,90) for s in a.severities):
        raise ValueError('Use non-empty severities from 20 40 60 90')
    cp=Path(a.checkpoint).expanduser().resolve()
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    if str(ck.get('version',''))!='2.0.0':
        raise RuntimeError('PCFT v2.0 evaluator requires a v2.0 checkpoint')
    saved_cfg=relocate_saved_config(ck['config'])
    if a.target and a.target!=saved_cfg['experiment']['target']:
        raise RuntimeError('Target does not match saved checkpoint')
    os.environ['TRAFFIDENT_TARGET']=saved_cfg['experiment']['target']
    cfg=safe_evaluation_config(saved_cfg, batch_size=a.batch_size,
        num_workers=a.num_workers,preload_repository_gpu=a.preload_repository_gpu)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Checkpoint-embedded Backbone path wins over an unrelated qsub -V environment.
    # Only an explicit CLI override may relocate the same SHA256-identical file.
    backbone_path=Path(a.backbone_checkpoint or ck['backbone_checkpoint']).expanduser()
    if not backbone_path.is_absolute():
        backbone_path=Path(cfg['project']['root'])/backbone_path
    if not backbone_path.is_file():
        raise FileNotFoundError(f'STLA checkpoint unavailable: {backbone_path}; pass --backbone-checkpoint')
    if ck.get('backbone_sha256') and file_sha256(backbone_path)!=ck['backbone_sha256']:
        raise RuntimeError('Incorrect STLA Backbone checkpoint: SHA256 differs from training')
    print(f'EVAL_MEMORY_SAFE device={dev} batch_size={cfg["training"]["batch_size"]} '
          f'num_workers={cfg["runtime"]["num_workers"]} '
          f'repository_gpu={cfg["runtime"]["preload_repository_gpu"]}', flush=True)
    model,_=build_model(cfg,backbone_path,dev)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval()
    mode=a.evidence_mode or saved_cfg['training'].get('ablation','full')
    store_path=cp.parent/'best_memory_index.npz'
    if mode!='no_retrieval':
        if not store_path.is_file():
            raise FileNotFoundError(f'Index must accompany checkpoint: {store_path}')
        index=LatentMemoryIndex.load(store_path).to(dev)
    else:
        index=None
    repo=FullTrainRepository(cache_path(cfg,'formal'))
    if cfg['runtime']['preload_repository_gpu']:
        repo.preload(dev)
    if index is not None and (
        len(repo)!=len(index.starts_np) or not np.array_equal(repo.starts,index.starts_np)
    ):
        raise RuntimeError('Index and Train-only repository fingerprints differ')
    mean=float(ck['mean']);std=float(ck['std'])
    rows=[]
    for sev in a.severities:
        ds=SupportSplit(cache_path(cfg,'evaluation',sev),'test')
        pr,used_batch=predict_with_retry(
            model,ds,index,repo,dev,mean,std,mode,
            bool(cfg['runtime'].get('amp',False)),cfg,make_loader,predict,
            cfg['training']['batch_size'])
        fm=summarize_by_family(pr['prediction'],pr['true'],pr['valid'],pr['direct'],ds.meta)
        bm=summarize_by_family(pr['base'],pr['true'],pr['valid'],pr['direct'],ds.meta)
        out=cp.parent/'evaluation'/f's{sev}'/mode
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'test_predictions.npz',
            prediction=pr['prediction'].astype(np.float32),
            base_prediction=pr['base'].astype(np.float32),
            correction=pr['correction'].astype(np.float32))
        save_summary(out/'summary.json',{
            'severity':sev,'mode':mode,'checkpoint_sha256':file_sha256(cp),
            'metrics':fm,'backbone':bm,'gate_mean':pr['gate_mean'],
            'useful_candidate_recall':pr['useful_candidate_recall'],
            'weighted_candidate_utility_normalized':pr['weighted_candidate_utility_normalized'],
            'evaluation_batch_size':used_batch,
            'evaluation_repository_gpu':repo._device is not None,
        })
        rows.append({'severity':sev,'mode':mode,**fm['ALL'],
                     **{'base_'+k:v for k,v in bm['ALL'].items()}})
        print('EVAL '+json.dumps(rows[-1]),flush=True)
        print(f'EVAL_SEVERITY_PASS s{sev} samples={len(ds)} batch={used_batch}',flush=True)
    edir=cp.parent/'evaluation'
    edir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(edir/f'severity_summary.{mode}.csv',index=False)
    if mode==saved_cfg['training'].get('ablation','full'):
        pd.DataFrame(rows).to_csv(edir/'severity_summary.csv',index=False)
        sfile=cp.parent/'summary.json'
        data=json.loads(sfile.read_text()) if sfile.is_file() else {}
        data.update(test_evaluated=True,evaluation_summary=rows)
        save_summary(sfile,data)
    print('EVALUATION_PASS',edir/f'severity_summary.{mode}.csv',flush=True)

if __name__=='__main__':
    main()
