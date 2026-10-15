#!/usr/bin/env python3
import _bootstrap,argparse,torch
from fare.config import read_config
from fare.pipeline import run

def cli():
    p=argparse.ArgumentParser(description='FARE v4 full single-model train/evaluate/analyze')
    p.add_argument('--config',default='FARE/configs/pilot.yaml')
    p.add_argument('--dataset');p.add_argument('--target',choices=['speed','flow']);p.add_argument('--seed',type=int)
    p.add_argument('--checkpoint',help='External frozen STLAformer best.pt')
    p.add_argument('--memory',help='Prebuilt FARE v4 episode Memory directory')
    p.add_argument('--oos-forecast-file',help='Explicit external OOF prediction .npz (formal)')
    p.add_argument('--run-dir');p.add_argument('--resume',help='Resume from FARE v4 last.pt')
    p.add_argument('--device');p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--skip-analysis',action='store_true')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--eval-val',action='store_true',help='Also evaluate/analyze validation after training; by default only test severities')
    g.add_argument('--test-only',action='store_true',help='Explicitly disable post-training validation evaluation')
    args=p.parse_args()
    cfg=read_config(args.config,args.dataset,args.target,args.seed,args.smoke)
    if args.eval_val:cfg['evaluation']['include_val']=True
    if args.test_only:cfg['evaluation']['include_val']=False
    device=torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    run(cfg,device,run_dir=args.run_dir,backbone_file=args.checkpoint,memory_file=args.memory,
        oos_forecast_file=args.oos_forecast_file,resume=args.resume,train=not args.evaluate_only,
        eval_only=args.evaluate_only,analysis=not args.skip_analysis)
if __name__=='__main__':cli()
