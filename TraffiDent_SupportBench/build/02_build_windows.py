#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from supportbench.config import load_config
from supportbench.splits import filter_complete_windows, valid_window_starts
from supportbench.utils import ensure_dir, log, read_json, relpath, uniform_temporal_subsample, write_json


def build_one(sg_dir: Path, cfg: dict):
    manifest=read_json(sg_dir/'manifest.json'); splits=read_json(sg_dir/'splits'/'time_splits.json')
    H=int(manifest['history_len']); P=int(manifest['horizon']); speed_idx=int(manifest['primary_channel_index'])
    valid=np.load(sg_dir/'core'/'raw_valid_mask.npy', mmap_mode='r')[...,speed_idx].astype(bool)
    out=ensure_dir(sg_dir/'windows'); files={}

    # Canonical FULL benchmark is dense for every split. Development budgets are
    # deterministic uniform temporal subsets of these dense canonical windows.
    for split,stride_key in [('train','train_stride'),('val','val_stride'),('test','test_stride')]:
        r=splits[split]
        stride=int(cfg['forecast'][stride_key])
        starts=valid_window_starts(r['start'],r['end_exclusive'],H,P,stride)
        starts=filter_complete_windows(starts,valid,H,P,float(cfg['forecast']['controlled_complete_min_fraction']))
        path=out/f'{split}_full.npy'; np.save(path,starts); files[f'{split}_full']=relpath(path,sg_dir)
        log(f'{sg_dir.name} {split}: {len(starts):,} dense controlled-complete windows (stride={stride})')

    budgets=cfg['budgets']
    for split,key in [('train','development_train_windows'),('val','development_val_windows'),('test','development_test_windows')]:
        base=np.load(out/f'{split}_full.npy')
        sub=uniform_temporal_subsample(base,int(budgets[key]))
        path=out/f'{split}_development.npy'; np.save(path,sub); files[f'{split}_development']=relpath(path,sg_dir)
        log(f'{sg_dir.name} {split}: development windows={len(sub):,}/{len(base):,}')

    rows=[]
    for split in ['train','val','test']:
        for start in np.load(out/f'{split}_full.npy'):
            start=int(start)
            rows.append({
                'window_id':f'{sg_dir.name}__w{start:09d}','subgraph':sg_dir.name,'split':split,'window_start':start,
                'history_end_exclusive':start+H,'future_end_exclusive':start+H+P,
            })
    pd.DataFrame(rows).to_csv(out/'window_manifest.csv',index=False)
    files['window_manifest']=relpath(out/'window_manifest.csv',sg_dir)
    sm={
        'splits':splits,'window_files':files,'history_len':H,'horizon':P,
        'selection':'controlled-complete underlying speed windows; full=5-min dense, development=uniform temporal subset',
        'canonical_strides':{s:int(cfg['forecast'][f'{s}_stride']) for s in ['train','val','test']},
        'development_budgets':{s:int(budgets[f'development_{s}_windows']) for s in ['train','val','test']},
    }
    write_json(sg_dir/'splits'/'split_manifest.json',sm)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',action='append',default=[]); args=ap.parse_args(); cfg=load_config(args.config)
    root=Path(cfg['project']['benchmark_root']); items=sorted(root.glob('TraffiDent_Net_*'))
    if args.subgraph: items=[x for x in items if x.name in set(args.subgraph)]
    for x in items: build_one(x,cfg)
if __name__=='__main__': main()
