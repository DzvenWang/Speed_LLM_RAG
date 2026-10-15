#!/usr/bin/env python3
"""Fail fast on data, checkpoint, graph and environment BEFORE GPU scheduling."""
import _bootstrap,argparse,json,importlib.util,torch
import numpy as np,pandas as pd
from fare.config import read_config,cache_dir,backbone_checkpoint,root
from fare.protocol import SupportDataset,graph_files,directed_metadata
from fare.traffic import TrafficGraph

def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='FARE/configs/pilot.yaml')
    p.add_argument('--dataset');p.add_argument('--target',choices=('speed','flow'))
    p.add_argument('--checkpoint');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();cfg=read_config(a.config,a.dataset,a.target,smoke=a.smoke)
    if importlib.util.find_spec('pandas.core.reshape.reshape') is None:
        raise RuntimeError('Pandas reshape module is missing: reinstall pandas cleanly')
    _=pd.crosstab(pd.Series(['a','b']),pd.Series(['c','c']))
    cache=cache_dir(cfg);train=SupportDataset(cache,'train',1)
    adjacency=graph_files(cache)['adj']
    if adjacency.shape!=(train.n,train.n):raise RuntimeError('Adjacency size mismatch')
    directed=directed_metadata(cfg['experiment']['dataset'],cfg['experiment']['target'],root())
    graph=TrafficGraph(adjacency,directed,cfg['traffic']['region_size'],
                       cfg['traffic']['neighbor_count'],cfg['traffic'].get('max_hops',3),
                       cfg['traffic'].get('path_sources',16))
    ckpath=backbone_checkpoint(cfg,a.checkpoint)
    payload=torch.load(ckpath,map_location='cpu',weights_only=False)
    if payload.get('model')!='STLAformer' or int(payload.get('num_nodes',-1))!=train.n:
        raise RuntimeError('Checkpoint model/node mismatch')
    report={'dataset':cfg['experiment']['dataset'],'target':cfg['experiment']['target'],
            'n_nodes':train.n,'history':train.h,'horizon':train.p,
            'cache':str(cache),'backbone_checkpoint':str(ckpath),
            'graph':graph.metadata_source,'directional_links':graph.directed_edges,
            'sparse_path_slots':int(graph.path_valid.sum()),
            'residual_mode':cfg['memory']['residual_mode'],
            'formal_requires_oos':cfg['experiment']['run_tag']=='formal',
            'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__}
    if cfg['experiment']['run_tag']=='formal' and cfg['memory']['residual_mode']!='external_oos':
        raise RuntimeError('Formal cannot use in-sample residuals')
    print('FARE4_PREFLIGHT_PASS '+json.dumps(report),flush=True)
if __name__=='__main__':cli()
