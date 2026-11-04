#!/usr/bin/env python3
from __future__ import annotations
import sys, argparse, os
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from pcft_stla.v6.timestamp import install
install()
from pcft_stla.v6.config import load_config
from pcft_stla.v6.pipeline import run

def main():
    p=argparse.ArgumentParser(description='PCFT v6: train all four stages, then final test and validation interventions')
    p.add_argument('--config',default=str(HERE/'configs/pcft_v6.yaml'))
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--run-name')
    p.add_argument('--resume',help='Path to last.pt; must match config/data/backbone')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--workers',type=int)
    a=p.parse_args()
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config,target=a.target,dataset=a.dataset,seed=a.seed)
    if a.batch_size:cfg['training']['batch_size']=a.batch_size;cfg['training']['auto_batch_probe']=False
    if a.workers is not None:cfg['runtime']['workers']=a.workers
    run(cfg,checkpoint=a.backbone_checkpoint,run_name=a.run_name,smoke=a.smoke,resume=a.resume,
        preflight_only=a.preflight_only)

if __name__=='__main__':main()
