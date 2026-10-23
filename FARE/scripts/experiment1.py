#!/usr/bin/env python3
"""FARE v6.2 Experiment 1 -- fresh FARE support, immutable candidates, T0-T3.

Run on GPU compute nodes, never use Test results to choose a Transfer method.
"""
import _bootstrap
import argparse,json
from pathlib import Path
import torch
from fare.config import read_config,project_path,root,dump_json
from fare.experiment1_v62 import train_support,candidate_bank,direct_feasibility,train_transfer,summarize_experiment,VARIANTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,choices=('speed','flow'))
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--stage',required=True,choices=('foundation','support','candidate_bank','feasibility','transfer','assess','all'))
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--config',default='FARE/configs/experiment1.yaml')
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--memory')
    p.add_argument('--experiment-dir')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.stage=='transfer' and not args.variant:raise ValueError('--variant is required for --stage transfer')
    cfg=read_config(args.config,args.dataset,args.target,args.seed)
    cfg['experiment']['train_limit']=None;cfg['experiment']['val_limit']=None
    if args.smoke:
        cfg['experiment1'].update({'support_epochs':1,'transfer_epochs':1,
           'pilot_train_limit':12,'pilot_val_limit':8,'workers':0,
           'support_batch_size':2,'candidate_batch_size':2,'transfer_batch_size':2})
        cfg['experiment']['train_limit']=12;cfg['experiment']['val_limit']=8
    basedir=(project_path(args.experiment_dir) if args.experiment_dir else
             root()/'FARE_exps'/args.target/args.dataset/'v62_experiment1'/(
               f'seed{args.seed}'+('_smoke' if args.smoke else '')))
    if args.stage=='assess':
        if not basedir.exists():raise FileNotFoundError(basedir)
        summarize_experiment(cfg,basedir)
        return
    from fare.config import backbone_checkpoint as resolve_backbone
    support_summary=basedir/'support'/'support_summary.json'
    summary=json.loads(support_summary.read_text()) if support_summary.exists() else {}
    backbone=resolve_backbone(cfg,args.backbone_checkpoint or summary.get('backbone_checkpoint'))
    if args.memory or summary.get('memory_path'):
        memory=project_path(args.memory or summary['memory_path'],must_exist=True)
    else:
        memory=root()/'FARE_cache'/args.target/args.dataset/f'memory_v502_{args.seed}_{cfg["memory"]["max_episodes"]}'
    if not (memory/'manifest.json').is_file():raise FileNotFoundError(memory/'manifest.json')
    basedir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('FARE62_EXPERIMENT1_START '+json.dumps({'stage':args.stage,'variant':args.variant,
      'target':args.target,'dataset':args.dataset,'fresh_initialization':True,'reuse_only':'OOF + original frozen STLAformer',
      'support_checkpoint':str(basedir/'support'/'support_best.pt'),'device':str(device),'smoke':args.smoke,
      'backbone':str(backbone),'memory':str(memory),'output':str(basedir)}),flush=True)
    if args.stage in ('support','foundation','all'):
        train_support(cfg,device,backbone,memory,basedir)
    if args.stage in ('candidate_bank','foundation','all'):
        candidate_bank(cfg,device,backbone,memory,basedir)
    if args.stage in ('foundation','feasibility','all'):
        direct_feasibility(cfg,device,backbone,memory,basedir)
    if args.stage in ('transfer','all'):
        for variant in ((args.variant,) if args.variant else VARIANTS):
            train_transfer(cfg,device,backbone,memory,basedir,variant)
    if args.stage=='all':summarize_experiment(cfg,basedir)
    print('FARE62_EXPERIMENT1_STAGE_PASS '+json.dumps({'stage':args.stage,'output':str(basedir)}),flush=True)

if __name__=='__main__':main()
