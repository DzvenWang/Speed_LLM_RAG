#!/usr/bin/env python3
import _bootstrap,argparse,json,torch
from pathlib import Path
from fare.config import read_config,cache_dir,backbone_checkpoint,root
from fare.protocol import SupportDataset,graph_files,directed_metadata

def cli():
    p=argparse.ArgumentParser();p.add_argument('--config',default='FARE/configs/pilot.yaml')
    p.add_argument('--dataset');p.add_argument('--target');p.add_argument('--checkpoint');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();c=read_config(a.config,a.dataset,a.target,smoke=a.smoke)
    path=cache_dir(c);ds=SupportDataset(path,'train',1)
    graph=graph_files(path);backbone=backbone_checkpoint(c,a.checkpoint)
    directed=directed_metadata(c['experiment']['dataset'],c['experiment']['target'],root())
    if graph['adj'].shape[0]!=ds.n:raise RuntimeError('Graph size mismatch')
    out={'dataset':c['experiment']['dataset'],'target':c['experiment']['target'],'nodes':ds.n,
         'train_rows':len(ds.meta),'backbone_checkpoint':str(backbone),
         'directional_metadata':'available' if directed is not None else 'unavailable (undirected proxy only)',
         'residual_mode':c['memory']['residual_mode']}
    print('FARE_V3_PREFLIGHT_PASS '+json.dumps(out),flush=True)
if __name__=='__main__':cli()
