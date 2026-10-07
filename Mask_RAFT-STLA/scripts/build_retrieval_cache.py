#!/usr/bin/env python3
from pathlib import Path
import argparse, os, sys, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import load_config, cache_path
from mask_raft_stla.retrieval import build_retrieval_cache

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target',choices=['speed','flow']); ap.add_argument('--split',choices=['train','val'],default='train'); ap.add_argument('--severity',type=int,choices=[20,40,60,90]); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--force',action='store_true'); ap.add_argument('--config',default=None); a=ap.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config); stage='evaluation' if a.severity is not None else 'formal'; split='test' if stage=='evaluation' else a.split; cache=cache_path(cfg,stage,a.severity); memory_cache=cache_path(cfg,'formal')
    tag=f'eval_s{a.severity}' if a.severity is not None else ('formal_smoke' if a.smoke else 'formal'); out=Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/tag/f'{split}.mask_raft_topk.npz'
    if out.exists() and not a.force: print(f'RETRIEVAL_CACHE_EXISTS {out}'); return
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    build_retrieval_cache(cache,split,out,cfg['retrieval']['periods'],int(cfg['retrieval']['top_k']),float(cfg['retrieval']['coverage_gamma']),8 if a.smoke else int(cfg['retrieval']['query_batch_size']),256 if a.smoke else int(cfg['retrieval']['memory_chunk_size']),int(cfg['retrieval']['min_valid_features']),dev,limit=32 if a.smoke else None,memory_cache_dir=memory_cache,memory_limit=512 if a.smoke else None)
if __name__=='__main__': main()
