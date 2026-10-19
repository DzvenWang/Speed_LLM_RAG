#!/usr/bin/env python3
import _bootstrap,argparse,torch
from pathlib import Path
from fare.config import project_path
from fare.pipeline import prepare,post_training_datasets
from fare.evaluation import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--memory');p.add_argument('--out')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--eval-val',action='store_true',help='Include Val in post-training evaluation/analysis')
    g.add_argument('--test-only',action='store_true',help='Evaluate/analyze Test severities only')
    a=p.parse_args();ck=torch.load(project_path(a.checkpoint,must_exist=True),map_location='cpu',weights_only=False)
    cfg=ck['config'];meta=ck['metadata'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.setdefault('evaluation',{}).setdefault('include_val',False)
    if a.eval_val:cfg['evaluation']['include_val']=True
    if a.test_only:cfg['evaluation']['include_val']=False
    model,memory,train,val,_,current=prepare(cfg,device,meta['backbone_checkpoint'],project_path(a.memory) if a.memory else meta['memory_path'])
    if meta['backbone_sha256']!=current['backbone_sha256'] or meta['memory_sha256']!=current['memory_sha256']:
        raise RuntimeError('Checkpoint dependency drift')
    model.load_state_dict(ck['state_dict'],strict=True)
    for name,ds in post_training_datasets(cfg,val).items():
        result,_=evaluate(model,memory,ds,device,cfg,(project_path(a.out) if a.out else project_path(a.checkpoint).parent/'evaluation')/name)
        print('FARE_V5_EVALUATION',name,result.get('overall_MAE'),result.get('overall_RMSE'),flush=True)
if __name__=='__main__':main()
