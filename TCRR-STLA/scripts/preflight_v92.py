#!/usr/bin/env python3
"""Fast v9.2 path/protocol preflight; no expensive teacher or replay computation."""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--physical-graph',default=None)
    p.add_argument('--node-ids',default=None)
    p.add_argument('--require-gpu',action='store_true')
    a=p.parse_args()
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    cfg['experiment'].update(target=a.target,dataset=a.dataset)
    ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    data=cache_path(cfg,'formal')
    files=['train_x_full_norm.npy','train_y_norm.npy','train_history_observed.npy',
           'train_artificial_missing.npy','train_manifest.csv','val_x_obs_norm.npy',
           'val_history_observed.npy','val_y_norm.npy']
    missing=[v for v in files if not (data/v).is_file()]
    if missing:raise FileNotFoundError('Missing required cache: '+', '.join(missing))
    full=np.load(data/'train_x_full_norm.npy',mmap_mode='r')
    mask=np.load(data/'train_history_observed.npy',mmap_mode='r')
    y=np.load(data/'train_y_norm.npy',mmap_mode='r')
    if full.ndim!=3 or full.shape!=mask.shape or len(y)!=len(full) or full.shape[1]!=12 or y.shape[1]!=12:
        raise RuntimeError('Incompatible TCRR v9.2 history/horizon/data shapes')
    if a.require_gpu and not torch.cuda.is_available():raise RuntimeError('CUDA required for GPU PBS v9.2')
    vendor=Path(cfg['project']['root'])/'Backbone/vendor/STLAformer/model/STLformer.py'
    if not vendor.is_file():raise FileNotFoundError(f'Original STLAformer vendor missing: {vendor}')
    raw=torch.load(ck,map_location='cpu',weights_only=False)
    if raw.get('model')!='STLAformer' or raw.get('num_nodes')!=full.shape[-1]:
        raise RuntimeError('Checkpoint model/nodes mismatch')
    if float(raw['std'])<=0:raise ValueError('Invalid standard deviation')
    if a.physical_graph:
        if not a.node_ids:raise ValueError('Physical graph requires --node-ids')
        with np.load(a.physical_graph,allow_pickle=False) as z:
            ids=np.load(a.node_ids,allow_pickle=False)
            if 'node_ids' not in z or not np.array_equal(z['node_ids'].astype(str),ids.astype(str)):
                raise ValueError('Graph node ids not aligned with dataset order')
    print('V92_PREFLIGHT_PASS '+json.dumps(dict(target=a.target,dataset=a.dataset,
        rows=len(full),nodes=full.shape[-1],history=full.shape[1],horizon=y.shape[1],
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        device_mem_gib=(torch.cuda.get_device_properties(0).total_memory/2**30 if torch.cuda.is_available() else None),
        physical_graph=bool(a.physical_graph),original_checkpoint=str(ck))),flush=True)
if __name__=='__main__':main()
