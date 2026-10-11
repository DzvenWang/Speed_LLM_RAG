from __future__ import annotations
import time,os,json
from pathlib import Path
import torch
from .config import cache_path,backbone_checkpoint,sha256,json_write,project_root
from .backbone import load_pretrained_stla
from .data import SupportSplit,graph_from_cache,directed_from_benchmark
from .memory import TrainMemory,populate_latents
from .model import FARE

def make_run_dir(cfg,tag='formal',stamp=None):
    ex=cfg['experiment'];s=stamp or time.strftime('%Y%m%d_%H%M%S')
    return project_root()/'FARE_exps'/ex['target']/ex['dataset']/tag/f'{s}_seed{ex["seed"]}'

def setup(cfg,ckpt_path=None,memory_path=None,device=None,smoke=False):
    device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    cache=cache_path(cfg,'formal')
    train_strategy=cfg['experiment'].get('train_sample_strategy','head')
    limit=cfg['experiment'].get('max_train_rows')
    if train_strategy=='spread' and limit is not None:
        from torch.utils.data import Subset
        import numpy as np
        base_train=SupportSplit(cache,'train')
        count=min(int(limit),len(base_train))
        if count<1:raise ValueError('Pilot train max_train_rows must be positive')
        idx=np.linspace(0,len(base_train)-1,count,dtype=np.int64)
        train=Subset(base_train,idx.tolist())
        train.num_nodes=base_train.num_nodes
    elif train_strategy=='head':
        train=SupportSplit(cache,'train',limit)
    else:
        raise ValueError('train_sample_strategy must be head or spread')
    val=SupportSplit(cache,'val',cfg['experiment'].get('max_val_rows'))
    if train.num_nodes!=val.num_nodes:raise RuntimeError('train / val node count mismatch')
    graph=graph_from_cache(cache)
    if graph.shape[0]!=train.num_nodes:raise RuntimeError('Graph / dataset shape mismatch')
    checkpoint=backbone_checkpoint(cfg,ckpt_path)
    pretrained,ck=load_pretrained_stla(checkpoint,device)
    if int(ck['num_nodes'])!=train.num_nodes:raise RuntimeError(f'Checkpoint expects {ck["num_nodes"]} nodes, cache has {train.num_nodes}')
    directed=directed_from_benchmark(cfg['experiment']['dataset'],cfg['experiment']['target'])
    model=FARE(pretrained,graph.to(device),cfg,directed=directed).to(device)
    if memory_path:
        memory=TrainMemory.load(memory_path,device=device)
        import json
        path=Path(memory_path).with_suffix('.json')
        if path.exists():
            stored=json.loads(path.read_text(encoding='utf-8'))
            for field,value in [('dataset',cfg['experiment']['dataset']),('target',cfg['experiment']['target'])]:
                if stored.get(field) and stored[field]!=value: raise RuntimeError(f'Memory {field} mismatch: {stored[field]} != {value}')
    else:
        memory=TrainMemory.from_cache(
            cache,(1 if cfg['experiment'].get('training_mode')=='no_memory' else cfg['memory'].get('max_rows')),
            device=device,sampling=cfg['memory'].get('sampling','head')
        )
    if memory.x.shape[-1]!=train.num_nodes:raise RuntimeError('Memory / network mismatch')
    if cfg.get('forecast_utility',{}).get('enabled',False) and cfg['experiment'].get('training_mode')!='no_memory' and memory.future is None:
        raise RuntimeError('FARE v2 needs historical-future TRAIN memory; rebuild it with v2 build_memory.py')
    if cfg['experiment'].get('training_mode')!='no_memory':
        populate_latents(memory,model,dim=cfg['innovation']['memory_dim'])
    meta={'dataset':cfg['experiment']['dataset'],'target':cfg['experiment']['target'],'mean':float(ck['mean']),'std':float(ck['std']),'backbone_checkpoint':str(checkpoint),'backbone_sha256':sha256(checkpoint),'cache':str(cache),'num_nodes':train.num_nodes}
    return model,memory,train,val,device,meta
