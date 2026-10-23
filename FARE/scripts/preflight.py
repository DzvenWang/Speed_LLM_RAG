#!/usr/bin/env python3
"""Fail fast on data, checkpoint, graph and environment BEFORE GPU scheduling."""
import _bootstrap,argparse,json,importlib.util,time,torch
import numpy as np,pandas as pd
from fare.config import read_config,cache_dir,backbone_checkpoint,root
from fare.protocol import SupportDataset,graph_files,directed_metadata
from fare.traffic import TrafficGraph

def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='FARE/configs/formal.yaml')
    p.add_argument('--dataset');p.add_argument('--target',choices=('speed','flow'))
    p.add_argument('--checkpoint');p.add_argument('--smoke',action='store_true')
    p.add_argument('--oos-forecast-file',help='Required for a new FARE v5 Memory unless already built')
    p.add_argument('--memory',help='Existing FARE v5 Memory dir')
    a=p.parse_args();cfg=read_config(a.config,a.dataset,a.target,smoke=a.smoke)
    start_time=time.monotonic()
    print('FARE6_PREFLIGHT_START low_memory_mode=1',flush=True)
    if importlib.util.find_spec('pandas.core.reshape.reshape') is None:
        raise RuntimeError('Pandas reshape module is missing: reinstall pandas cleanly')
    _=pd.crosstab(pd.Series(['a','b']),pd.Series(['c','c']))
    print('FARE6_PREFLIGHT_CACHE_BEGIN',flush=True)
    cache=cache_dir(cfg);train=SupportDataset(cache,'train',1)
    print(f'FARE6_PREFLIGHT_CACHE_PASS nodes={train.n}',flush=True)
    adjacency=graph_files(cache)['adj']
    if adjacency.shape!=(train.n,train.n):raise RuntimeError('Adjacency size mismatch')
    directed=directed_metadata(cfg['experiment']['dataset'],cfg['experiment']['target'],root())
    graph=TrafficGraph(adjacency,directed,cfg['traffic']['region_size'],
                       cfg['traffic']['neighbor_count'],cfg['traffic'].get('max_hops',3),
                       cfg['traffic'].get('path_sources',16))
    print('FARE6_PREFLIGHT_GRAPH_PASS',flush=True)
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
    if cfg['experiment']['run_tag']=='formal' and cfg['memory']['residual_mode']!='external_oos_paired':
        raise RuntimeError('Formal cannot use in-sample residuals')
    print('FARE6_PREFLIGHT_CHECKPOINT_PASS',flush=True)
    # Fail closed BEFORE submitting an expensive Gadi job.
    from pathlib import Path
    from fare.config import project_path,hash_file
    from fare.memory_preflight import inspect_cached_memory
    if cfg['memory']['residual_mode']=='external_oos_paired':
        memory_dir=project_path(a.memory) if a.memory else root()/'FARE_cache'/cfg['experiment']['target']/cfg['experiment']['dataset']/f"memory_v502_{cfg['experiment']['seed']}_{cfg['memory']['max_episodes']}"
        if a.memory or (memory_dir/'manifest.json').exists():
            print('FARE6_PREFLIGHT_MEMORY_HEADERS_BEGIN (no tensor materialization)',flush=True)
            meta=inspect_cached_memory(memory_dir,nodes=train.n,history=train.h,horizon=train.p,
                                      expected={'dataset':cfg['experiment']['dataset'],
                                                'target':cfg['experiment']['target'],
                                                'backbone_sha256':hash_file(ckpath)})
            print(f'FARE6_PREFLIGHT_MEMORY_HEADERS_PASS episodes={meta["episodes"]}',flush=True)
            if a.oos_forecast_file:
                print('FARE6_PREFLIGHT_OOF_SHA256_BEGIN (streaming)',flush=True)
                actual=hash_file(project_path(a.oos_forecast_file,must_exist=True))
                if meta.get('oos_source_sha256')!=actual:
                    raise RuntimeError('Cached OOF Memory was built from a DIFFERENT OOF source; rebuild Memory before qsub')
                print('FARE6_PREFLIGHT_OOF_SHA256_PASS',flush=True)
            report['memory']=str(memory_dir)
        elif a.oos_forecast_file:
            from fare.oof import verify_paired
            print('FARE6_PREFLIGHT_OOF_VERIFY_BEGIN (no cached memory)',flush=True)
            src=verify_paired(project_path(a.oos_forecast_file,must_exist=True),train,strict=True)
            report['oof_rows']=len(src['row_id'])
            report['oof_folds']=int(len(np.unique(src['fold_id'])))
            report['oof_file']=str(project_path(a.oos_forecast_file,must_exist=True))
        else:
            raise RuntimeError('Missing paired OOF source and v5 Memory; create external OOF first')
    report['elapsed_seconds']=round(time.monotonic()-start_time,2)
    print('FARE5_PREFLIGHT_PASS '+json.dumps(report),flush=True)
if __name__=='__main__':cli()
