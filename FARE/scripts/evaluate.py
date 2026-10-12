#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
import _bootstrap
from fare.config import config_from_checkpoint,cache_path
from fare.runtime import setup
from fare.trainer import load_ckpt
from fare.data import SupportSplit
from fare.evaluation import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--dataset');p.add_argument('--target');p.add_argument('--memory');p.add_argument('--config')
    p.add_argument('--mode',default=None,choices=['full','retrieval_off','state_only','innovation_only','global_retrieval','no_mask_alignment','random_memory','no_risk','no_utility','constant_alpha','no_memory','no_learned_reranker','no_observation_distance','utility_off','similarity_memory']);p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--severities',type=int,nargs='+',default=[20,40,60,90]);p.add_argument('--batch-size',type=int);p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    exp=ck['config']['experiment'];dataset=args.dataset or exp['dataset'];target=args.target or exp['target']
    if dataset!=exp['dataset'] or target!=exp['target']:raise ValueError('Checkpoint / evaluation dataset or target mismatch')
    if args.config:
        raise ValueError('Evaluation must use the checkpoint-stored configuration; --config overrides are unsafe')
    cfg=config_from_checkpoint(ck,smoke=args.smoke)
    if args.mode is None:
        args.mode=cfg['experiment'].get('training_mode','full')
    model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],args.memory or ck.get('memory_path'))
    if meta['backbone_sha256']!=ck['metadata']['backbone_sha256']:raise RuntimeError('Backbone source checkpoint changed')
    load_ckpt(args.checkpoint,model,device)
    folder=Path(args.output) if args.output else Path(args.checkpoint).parent/'evaluation'
    batch_size=args.batch_size or cfg['training']['batch_size']
    if args.split=='val':
        result=evaluate(model,memory,val,device,batch_size,meta['mean'],meta['std'],folder,'val_'+args.mode,True,mode=args.mode)
        print('FARE_STAGE_EVALUATION '+json.dumps(result['final']),flush=True)
        return
    for severity in args.severities:
        cache=cache_path(cfg,'evaluation',severity)
        ds=SupportSplit(cache,'test',16 if args.smoke else None)
        result=evaluate(model,memory,ds,device,batch_size,meta['mean'],meta['std'],folder/f's{severity:02d}',f's{severity:02d}_{args.mode}',True,mode=args.mode)
        print('FARE_TEST '+json.dumps({'dataset':dataset,'target':target,'severity':severity,**result['final']}),flush=True)
    print('FARE_EVALUATE_PASS',flush=True)
if __name__=='__main__':main()
