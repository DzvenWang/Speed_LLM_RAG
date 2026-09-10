#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from supportbench.config import load_config
from supportbench.memory import build_static_patches
from supportbench.utils import ensure_dir, log, read_json, write_json, uniform_temporal_subsample


def _spaced(starts: np.ndarray, stride: int) -> np.ndarray:
    starts=np.asarray(starts,np.int64)
    if not len(starts): return np.empty(0,np.int64)
    out=[int(starts[0])]; last=int(starts[0])
    for s in starts[1:]:
        s=int(s)
        if s-last>=int(stride): out.append(s); last=s
    return np.asarray(out,np.int64)


def build_one(sg_dir: Path, cfg: dict):
    man=read_json(sg_dir/'manifest.json'); H=int(man['history_len']); P=int(man['horizon'])
    stride=int(cfg['memory']['episode_stride']); out=ensure_dir(sg_dir/'memory')
    full=_spaced(np.load(sg_dir/'windows'/'train_full.npy'),stride)
    dev=uniform_temporal_subsample(full,int(cfg['budgets']['development_train_windows']))
    np.save(out/'episode_window_starts_full.npy',full); np.save(out/'episode_window_starts_development.npy',dev)
    # Backward-compatible alias points to formal/full memory, without duplicating data.
    np.save(out/'episode_window_starts.npy',full)
    devset=set(map(int,dev))
    rows=[]
    for i,s in enumerate(full):
        rows.append({'episode_id':i,'window_id':f'{sg_dir.name}__w{int(s):09d}','window_start':int(s),'history_end_exclusive':int(s)+H,'future_end_exclusive':int(s)+H+P,'in_development_memory':bool(int(s) in devset)})
    pd.DataFrame(rows).to_csv(out/'episodes.csv',index=False)
    adj=np.load(sg_dir/'graph'/'adjacency.npy',mmap_mode='r')
    ids,valid=build_static_patches(np.asarray(adj),int(cfg['memory']['local_patch_hops']),int(cfg['memory']['local_patch_max_nodes']))
    np.savez_compressed(out/'local_patches.npz',ids=ids,valid=valid)
    write_json(out/'memory_manifest.json',{
        'train_only':True,'episode_stride_raw_steps':stride,'history_len':H,'horizon':P,
        'episode_count_full':int(len(full)),'episode_count_development':int(len(dev)),
        'formal_memory_budget':str(cfg['memory'].get('formal_memory_budget','full')),
        'development_memory_budget':str(cfg['memory'].get('development_memory_budget','development')),
        'strict_temporal_exclusion_for_train_queries':bool(cfg['memory']['strict_temporal_exclusion']),
        'local_case_definition':'lazy Cartesian pair (episode_id,target_node); local patch is static per target node',
        'local_patch_hops':int(cfg['memory']['local_patch_hops']),'local_patch_max_nodes':int(cfg['memory']['local_patch_max_nodes']),
        'canonical_values':'historical future is read from core traffic tensor; no model embedding/residual is stored',
    })
    log(f'{sg_dir.name}: full memory episodes={len(full):,}; development memory episodes={len(dev):,}; stride={stride}')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',action='append',default=[]); args=ap.parse_args(); cfg=load_config(args.config)
    root=Path(cfg['project']['benchmark_root']); items=sorted(root.glob('TraffiDent_Net_*'))
    if args.subgraph: items=[x for x in items if x.name in set(args.subgraph)]
    for x in items: build_one(x,cfg)
if __name__=='__main__': main()
