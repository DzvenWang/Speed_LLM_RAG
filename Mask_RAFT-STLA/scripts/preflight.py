#!/usr/bin/env python3
"""Protocol and checkpoint sanity check before spending Gadi GPU time."""
import argparse,os,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import load_config,cache_path,resolve_stla_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['speed','flow']);p.add_argument('--dataset');p.add_argument('--config');p.add_argument('--backbone-checkpoint');a=p.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config)
    if a.dataset:cfg['experiment']['dataset']=a.dataset
    cache=cache_path(cfg,'formal');ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    required=['train_x_obs_norm.npy','train_x_full_norm.npy','train_history_observed.npy','train_y_norm.npy','train_y_raw.npy','train_future_valid.npy','train_manifest.csv','train_tod_hist.npy','train_dow_hist.npy',
       'val_x_obs_norm.npy','val_history_observed.npy','val_y_norm.npy','val_y_raw.npy','val_future_valid.npy','val_manifest.csv']
    absent=[f for f in required if not (cache/f).is_file()]
    if absent:raise RuntimeError('Missing required files: '+str(absent))
    model_source=Path(cfg['project']['root'])/'Backbone/vendor/STLAformer/model/STLformer.py'
    if not model_source.is_file():raise FileNotFoundError(model_source)
    state=torch.load(ck,map_location='cpu',weights_only=False)
    if state.get('model')!='STLAformer':raise RuntimeError(f'Not an STLAformer checkpoint: {ck}')
    x=np.load(cache/'train_x_full_norm.npy',mmap_mode='r');y=np.load(cache/'train_y_norm.npy',mmap_mode='r')
    if x.shape[2]!=int(state['num_nodes']):raise RuntimeError('STLA N does not match cache N')
    if x.shape[1]!=int(cfg['experiment']['history']) or y.shape[1]!=int(cfg['experiment']['horizon']):raise RuntimeError('HISTORY/HORIZON mismatch')
    if len(y)!=len(x):raise RuntimeError('Train future labels and histories have different rows')
    print(f'PREFLIGHT_PASS version={cfg["version"]} dataset={cfg["experiment"]["dataset"]} target={cfg["experiment"]["target"]} N={x.shape[2]} H={x.shape[1]} P={y.shape[1]} ckpt={ck} torch={torch.__version__}',flush=True)
if __name__=='__main__':main()
