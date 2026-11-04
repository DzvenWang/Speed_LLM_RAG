#!/usr/bin/env python3
"""Evaluate a finished v7 checkpoint on VAL or a declared test severity.

Interventions here are post-training only, NOT retrained independent ablations.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from pcft_stla.v6.pipeline import prepare
from pcft_stla.v6.model import load_original_stla
from pcft_stla.v7.pipeline import instantiate
from pcft_stla.v7.training import run_epoch
from pcft_stla.v6.config import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,help='PCFT v7 joint_best.pt')
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--split',default='val',choices=['val','test_s20','test_s40','test_s60','test_s90'])
    p.add_argument('--intervention',default='none',choices=['none','no_c1','no_c2','no_c3','no_local','no_direction','no_null'])
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if cp.get('schema')!='pcft_v7_best_v1':raise ValueError('Invalid v7 checkpoint format')
    cfg=cp['config'];target=cfg['experiment']['target'];dataset=cfg['experiment']['dataset']
    import os
    os.environ['TRAFFIDENT_TARGET']=target
    repo,_,base_path,graph,ds,identity=prepare(cfg,args.backbone_checkpoint,smoke=False)
    identity=dict(identity,v7_schema='pcft_v7_anchor_selective_v1',model_config=cfg['model'],
                  train_mask_mode=cfg['experiment']['train_mask_mode'])
    if cp['identity']!=identity:raise RuntimeError('Evaluation identity differs from training checkpoint')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=instantiate(cfg,repo,base_path,graph,identity,dev)
    model.load_state_dict(cp['model'],strict=True)
    model.anchor.set_trainable_tail(0)
    model.set_stage('joint')
    choices={'no_c1':{'use_support':False},'no_c2':{'use_propagation':False},
             'no_c3':{'use_controller':False},'no_local':{'use_local':False},
             'no_direction':{'use_direction':False},'no_null':{'use_null_source':False}}
    if args.intervention!='none':model.set_ablation(**choices[args.intervention])
    model.eval()
    cfg['runtime']['eval_batch']=args.batch_size
    result=run_epoch(model,ds[args.split],'joint',cfg,dev,float(identity['mean']),float(identity['std']))
    print('V7_EVALUATION',json.dumps({'split':args.split,'intervention':args.intervention,'checkpoint':str(args.checkpoint),
                                     'posthoc_intervention_not_retrained':True,'metrics':result},sort_keys=True),flush=True)

if __name__=='__main__':main()
