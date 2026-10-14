#!/usr/bin/env python3
"""Evaluate/analyze saved Full checkpoint, without updating weights."""
import _bootstrap,argparse,torch
from pathlib import Path
from fare.config import read_config
from fare.pipeline import prepare,post_training_datasets
from fare.analysis import analyze
from fare.protocol import SupportDataset
from fare.config import cache_dir

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--memory');p.add_argument('--out')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--eval-val',action='store_true',help='Include Val in post-training evaluation/analysis')
    g.add_argument('--test-only',action='store_true',help='Evaluate/analyze Test severities only')
    args=p.parse_args();ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config'];meta=ck['metadata'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.setdefault('evaluation',{}).setdefault('include_val',False)
    if a.eval_val:cfg['evaluation']['include_val']=True
    if a.test_only:cfg['evaluation']['include_val']=False
    model,memory,train,val,_,current=prepare(cfg,device,meta['backbone_checkpoint'],args.memory or meta['memory_path'])
    if meta['backbone_sha256']!=current['backbone_sha256'] or meta['memory_sha256']!=current['memory_sha256']:
        raise RuntimeError('Checkpoint dependency drift')
    model.load_state_dict(ck['state_dict'],strict=True)
    out=Path(args.out) if args.out else Path(args.checkpoint).parent/'analysis'
    analyze(model,memory,post_training_datasets(cfg,val),device,cfg,out)
if __name__=='__main__':main()
