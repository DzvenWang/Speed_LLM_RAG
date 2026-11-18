#!/usr/bin/env python3
"""CPU-only check that caches, STLA checkpoint and vendor code are consistent."""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from reshift2.config import load_config, cache_path, resolve_stla_checkpoint


def main():
    p = argparse.ArgumentParser(); p.add_argument('--target', choices=['speed', 'flow']); p.add_argument('--dataset')
    p.add_argument('--config'); p.add_argument('--backbone-checkpoint'); a = p.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET'] = a.target
    cfg = load_config(a.config)
    if a.dataset: cfg['experiment']['dataset'] = a.dataset
    cache = cache_path(cfg, 'formal'); ck = resolve_stla_checkpoint(cfg, a.backbone_checkpoint)
    need = ['train_x_obs_norm.npy', 'train_x_full_norm.npy', 'train_y_norm.npy', 'train_manifest.csv', 'train_tod_hist.npy',
            'train_dow_hist.npy', 'val_x_obs_norm.npy', 'val_y_norm.npy', 'val_y_raw.npy', 'val_history_observed.npy', 'val_manifest.csv']
    miss = [f for f in need if not (cache / f).is_file()]
    if miss: raise RuntimeError(f'Missing cache files: {miss}')
    for sev in cfg['experiment']['severities']: cache_path(cfg, 'evaluation', int(sev))
    state = torch.load(ck, map_location='cpu', weights_only=False)
    if state.get('model') != 'STLAformer': raise RuntimeError(f'Expected STLAformer checkpoint: {ck}')
    x = np.load(cache / 'train_x_full_norm.npy', mmap_mode='r'); y = np.load(cache / 'train_y_norm.npy', mmap_mode='r')
    if x.shape[2] != int(state['num_nodes']) or len(x) != len(y): raise RuntimeError('Backbone / cache dimension mismatch')
    print(f'RESHIFT_PREFLIGHT_PASS version={cfg["version"]} dataset={cfg["experiment"]["dataset"]} '
          f'target={cfg["experiment"]["target"]} N={x.shape[2]} rows={len(x)} checkpoint={ck}', flush=True)


if __name__ == '__main__': main()
