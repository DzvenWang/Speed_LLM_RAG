#!/usr/bin/env python3
"""Read-only dependency and protocol preflight; does not train or inspect TEST labels."""
from __future__ import annotations
import argparse,os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from corsta.observability import install_timestamps
install_timestamps()
from corsta.config import load_config,cache_path,resolve_stla_checkpoint
from corsta.data import SupportSplit
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target',required=True,choices=['speed','flow'])
    ap.add_argument('--dataset',default='TraffiDent_Net_06')
    ap.add_argument('--backbone-checkpoint',required=True)
    ap.add_argument('--skip-cpu-forward',action='store_true',help='Use only after a successful real CPU preflight')
    a=ap.parse_args()
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/bootstrap_v7.yaml')
    cfg['experiment'].update(target=a.target,dataset=a.dataset)
    ckpt=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    if ck.get('model')!='STLAformer':raise RuntimeError('Expected STLAformer checkpoint')
    for k,v in (('target',a.target),('dataset',a.dataset)):
        if ck.get(k) not in (None,v):raise RuntimeError(f'Backbone {k} mismatch')
    root=cache_path(cfg,'formal')
    train=SupportSplit(root,'train',limit=2,cf_training=True)
    val=SupportSplit(root,'val',limit=2)
    if train.x_full is None or train.original_valid is None:
        raise RuntimeError('Missing TRAIN full history or source validity')
    if train.x_obs.shape[-1]!=int(ck['num_nodes']) or val.x_obs.shape[-1]!=int(ck['num_nodes']):
        raise RuntimeError('Node count mismatch')
    required=['train_x_full_norm.npy','train_y_norm.npy','train_manifest.csv',
              'val_x_obs_norm.npy','val_manifest.csv']
    for file in required:
        if not (root/file).is_file():raise FileNotFoundError(root/file)
    # Detect data/model/mask shape mistakes while on the login node,
    # before an expensive gpuvolta job enters the queue. One tiny CPU batch.
    import numpy as np
    arr=train[0]
    observed=arr['mask'].bool(); source=arr['original_valid'].bool()
    if (observed & ~source).any():
        raise RuntimeError('TRAIN observed mask exposes a source-invalid measurement')
    if observed.any() and (arr['x'][observed]-arr['x_full_training_only'][observed]).abs().max()>1e-2:
        raise RuntimeError('TRAIN visible history differs from TRAIN full history')
    if not torch.isfinite(arr['x'][observed]).all() or not torch.isfinite(arr['y_norm'][arr['future_valid'].bool()]).all():
        raise RuntimeError('Nonfinite TRAIN observations or labels')
    if not np.isfinite(float(ck['mean'])) or not np.isfinite(float(ck['std'])) or float(ck['std'])<=0:
        raise RuntimeError('Invalid STLAformer normalization metadata')
    from corsta.v7.graph import graph_identity,_read_graph,_validate
    graph_identity(Path(cfg['project']['root']),a.target,a.dataset,root/'adjacency.npy',
                   int(cfg['model']['road_neighbors']))
    edges=_read_graph(Path(cfg['project']['root']),a.target,a.dataset,root/'adjacency.npy',
                   int(cfg['model']['road_neighbors']))
    _validate(edges,int(ck['num_nodes']),int(cfg['model']['road_neighbors']))
    if not a.skip_cpu_forward:
        from corsta.stla import load_pretrained_stla,MaskAwareSTLAformer
        torch.set_num_threads(min(4,torch.get_num_threads()))
        with torch.inference_mode():
            base,_=load_pretrained_stla(Path(cfg['project']['root']),ckpt,torch.device('cpu'))
            teacher=MaskAwareSTLAformer(base).eval()
            q=val[0]
            x=q['x'][None];mask=q['mask'][None]
            pred,latent=teacher(x,mask,q['tod'][None],q['dow'][None])
            expected_dim=int(base.model_dim)
            if pred.shape!=(1,12,int(ck['num_nodes'])) or latent.shape!=(1,int(ck['num_nodes']),expected_dim):
                raise RuntimeError(f'STLA interface mismatch prediction={pred.shape}, latent={latent.shape}')
            if not torch.isfinite(pred).all() or not torch.isfinite(latent).all():
                raise RuntimeError('STLA CPU forward nonfinite')
        print(f'V7_BACKBONE_CPU_FORWARD_PASS target={a.target} nodes={ck["num_nodes"]} dim={expected_dim}',flush=True)
    print('V7_PREFLIGHT_PASS '+json.dumps({'target':a.target,'dataset':a.dataset,
        'train_shape':list(train.x_obs.shape),'val_shape':list(val.x_obs.shape),
        'checkpoint':str(ckpt),'history_validity':str(root/'train_artificial_missing.npy')}),flush=True)

if __name__=='__main__':main()
