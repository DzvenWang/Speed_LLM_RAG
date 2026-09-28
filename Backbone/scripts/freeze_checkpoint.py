#!/usr/bin/env python3
"""Freeze exactly the checkpoint path explicitly selected by the user. No ranking/auto-selection."""
from pathlib import Path
import argparse, datetime, json, os, sys, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml, sha256_file
MODELS=['STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer']
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=MODELS)
    ap.add_argument('--target',required=True,choices=['speed','flow'])
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml'))
    a=ap.parse_args()
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_yaml(a.config)
    ckpt=Path(a.checkpoint).expanduser().resolve()
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict): raise RuntimeError('checkpoint must be a dict')
    got_model=str(ck.get('model') or '')
    if got_model and got_model != a.model: raise RuntimeError(f'checkpoint model={got_model} != requested {a.model}')
    got_seed=ck.get('seed')
    if got_seed is not None and int(got_seed)!=a.seed: raise RuntimeError(f'checkpoint seed={got_seed} != requested {a.seed}')
    sha=sha256_file(ckpt); exp=Path(cfg['project']['exp_root']); exp.mkdir(parents=True,exist_ok=True)
    run_identity=ck.get('run_identity') if isinstance(ck.get('run_identity'),dict) else {}
    entry={'checkpoint':str(ckpt),'checkpoint_sha256':sha,'run_dir':str(ckpt.parent),'best_epoch':ck.get('best_epoch'),'val_MAE':ck.get('best_val_MAE')}
    payload={'schema_version':5,'project':'Backbone','project_version':str(cfg['version']),'freeze_policy':'user_explicit_checkpoint_only','selection_mode':'user_explicit_checkpoint_path','selection_uses_test':False,'selected_model':a.model,'target':a.target,'reference_seed':a.seed,'reference_checkpoint':str(ckpt),'reference_checkpoint_sha256':sha,'reference_run_dir':str(ckpt.parent),'formal_seeds':[a.seed],'formal_checkpoints':{str(a.seed):entry},'run_identity':run_identity,'frozen_at':datetime.datetime.now().isoformat(timespec='seconds')}
    manifest=exp/'FROZEN_BACKBONE.json'
    manifest.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(f'FREEZE_PASS model={a.model} target={a.target} seed={a.seed} checkpoint={ckpt}')
    print(f'FROZEN_MANIFEST={manifest}')
if __name__=='__main__': main()
