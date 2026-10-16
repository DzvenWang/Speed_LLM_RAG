#!/usr/bin/env python3
from pathlib import Path
import argparse,os,sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pcft_stla.config import load_config,cache_path,resolve_stla_checkpoint
from pcft_stla.trainer import build_model,setup_graph
from pcft_stla.data import FullTrainRepository
from pcft_stla.memory import build_memory_index

def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['speed','flow']);p.add_argument('--dataset');p.add_argument('--config')
    p.add_argument('--backbone-checkpoint');p.add_argument('--smoke',action='store_true');p.add_argument('--force',action='store_true');a=p.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config)
    if a.dataset:cfg['experiment']['dataset']=a.dataset
    cache=cache_path(cfg,'formal');device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,_=build_model(cfg,resolve_stla_checkpoint(cfg,a.backbone_checkpoint),device)
    graph=setup_graph(model,cfg,cache,force=a.force)
    repo=FullTrainRepository(cache,256 if a.smoke else None)
    index=build_memory_index(model.memory_encoder,repo,device,16 if a.smoke else int(cfg['retrieval']['store_batch_size']),propagation=model.propagation if model.propagation_enabled else None)
    path=Path(cfg['project']['store_root'])/cfg['experiment']['dataset']/('initial_smoke_index.npz' if a.smoke else 'initial_index.npz')
    index.save(path,{'stage':'pretraining index only - formal model refreshes during training','rows':len(repo)})
    print(f'STORE_PASS graph={graph} index={path}',flush=True)
if __name__=='__main__':main()
