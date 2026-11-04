#!/usr/bin/env python3
from __future__ import annotations
import argparse,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from pcft_stla.v6.timestamp import install
install()
from pcft_stla.v6.config import load_config
from pcft_stla.v7.pipeline import run

def main():
    a=argparse.ArgumentParser(description='Unified PCFT v7 full model (STLA+C1+C2+C3)')
    a.add_argument('--config',default=str(HERE/'configs/pcft_v7.yaml'))
    a.add_argument('--target',required=True,choices=['speed','flow'])
    a.add_argument('--dataset',default='TraffiDent_Net_06')
    a.add_argument('--seed',type=int,default=10)
    a.add_argument('--backbone-checkpoint')
    a.add_argument('--run-name')
    a.add_argument('--resume')
    a.add_argument('--batch-size',type=int)
    a.add_argument('--train-mask-mode',choices=['fixed_s40','mixed_train_masks'])
    a.add_argument('--smoke',action='store_true')
    a.add_argument('--preflight-only',action='store_true')
    a.add_argument('--gpu-smoke-only',action='store_true')
    args=a.parse_args()
    os.environ['TRAFFIDENT_TARGET']=args.target
    cfg=load_config(args.config,target=args.target,dataset=args.dataset,seed=args.seed)
    if cfg['experiment']['train_mask_mode'] not in ('fixed_s40','mixed_train_masks'):
        raise ValueError('Unsupported train_mask_mode')
    if args.batch_size:cfg['training']['batch_size']=args.batch_size
    if args.train_mask_mode:cfg['experiment']['train_mask_mode']=args.train_mask_mode
    run(cfg,checkpoint=args.backbone_checkpoint,run_name=args.run_name,smoke=args.smoke,
        resume=args.resume,preflight_only=args.preflight_only,gpu_smoke_only=args.gpu_smoke_only)
if __name__=='__main__':main()
