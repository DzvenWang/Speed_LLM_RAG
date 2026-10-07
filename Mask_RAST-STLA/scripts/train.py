#!/usr/bin/env python3
from pathlib import Path
import argparse,datetime,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_rast_stla.config import load_config,cache_path,resolve_stla_checkpoint
from mask_rast_stla.trainer import train

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['speed','flow']);ap.add_argument('--seed',type=int,default=10);ap.add_argument('--backbone-checkpoint');ap.add_argument('--smoke',action='store_true');ap.add_argument('--run-stamp');ap.add_argument('--config',default=None);a=ap.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config);cfg['experiment']['seed']=int(a.seed);cache=cache_path(cfg,'formal');ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint);stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S');stage='formal_smoke' if a.smoke else 'formal';run=Path(cfg['project']['exp_root'])/cfg['experiment']['dataset']/stage/f'{stamp}_seed{a.seed}';best=train(cfg,cache,ck,run,a.smoke);print(f'BEST_CHECKPOINT={best}',flush=True)
if __name__=='__main__':main()
