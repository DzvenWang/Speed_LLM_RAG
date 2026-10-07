#!/usr/bin/env python3
from pathlib import Path
import argparse, os, sys, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_rast_stla.config import load_config,cache_path,resolve_stla_checkpoint

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['speed','flow']);ap.add_argument('--config',default=None);a=ap.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config);cache=cache_path(cfg,'formal');ck=resolve_stla_checkpoint(cfg);vendor=Path(cfg['project']['root'])/'Backbone'/'vendor'/'STLAformer'/'model'/'STLformer.py';required=['train_x_obs_norm.npy','train_x_full_norm.npy','train_history_observed.npy','train_y_raw.npy','train_tod_hist.npy','train_dow_hist.npy','train_manifest.csv','val_x_obs_norm.npy','val_history_observed.npy','val_y_raw.npy','val_manifest.csv'];missing=[x for x in required if not (cache/x).is_file()]
    if missing:raise SystemExit('Missing cache files: '+', '.join(missing))
    if not vendor.is_file():raise SystemExit(f'Missing STLAformer vendor source: {vendor}')
    c=torch.load(ck,map_location='cpu',weights_only=False);print(f'PREFLIGHT_PASS target={cfg["experiment"]["target"]} cache={cache} checkpoint={ck} N={c.get("num_nodes")} torch={torch.__version__}',flush=True)
if __name__=='__main__':main()
