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
    train=SupportSplit(cache,'train',cfg['experiment'].get('max_train_rows'))
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
        memory=TrainMemory.from_cache(cache,cfg['memory'].get('max_rows'),device=device)
    if memory.x.shape[-1]!=train.num_nodes:raise RuntimeError('Memory / network mismatch')
    populate_latents(memory,model,dim=cfg['innovation']['memory_dim'])
    meta={'dataset':cfg['experiment']['dataset'],'target':cfg['experiment']['target'],'mean':float(ck['mean']),'std':float(ck['std']),'backbone_checkpoint':str(checkpoint),'backbone_sha256':sha256(checkpoint),'cache':str(cache),'num_nodes':train.num_nodes}
    return model,memory,train,val,device,meta
