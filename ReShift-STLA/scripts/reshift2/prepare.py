#!/usr/bin/env python3
"""Build every shared cache of one target ONCE (graph, T(x_full), Train/Val/Test replay+CF banks).

Submit this first, then all runs of the target with  qsub -W depend=afterok:<job>  so that
no two jobs build the same bank.  Re-running is a no-op when everything is cached.
"""
from pathlib import Path
import argparse, os, sys, time
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from reshift2.config import load_config, cache_path, resolve_stla_checkpoint
from reshift2.data import SupportSplit, FullTrainRepository
from reshift2.trainer import build_model, setup_graph, query_bank, bank_store
from reshift2.teacher import prepare_pfull_bank


def main():
    p = argparse.ArgumentParser(); p.add_argument('--target', choices=['speed', 'flow']); p.add_argument('--dataset')
    p.add_argument('--config'); p.add_argument('--backbone-checkpoint'); a = p.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET'] = a.target
    cfg = load_config(a.config)
    if a.dataset: cfg['experiment']['dataset'] = a.dataset
    t0 = time.time(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = cache_path(cfg, 'formal'); ck = resolve_stla_checkpoint(cfg, a.backbone_checkpoint)
    model, _ = build_model(cfg, ck, dev); setup_graph(model, cfg, cache)
    repo = FullTrainRepository(cache); repo.preload(dev)
    pfull = prepare_pfull_bank(model.teacher, repo, ck, bank_store(cfg), dev)
    splits = [('train', SupportSplit(cache, 'train')), ('val', SupportSplit(cache, 'val'))]
    splits += [(f'test_s{s}', SupportSplit(cache_path(cfg, 'evaluation', int(s)), 'test')) for s in cfg['experiment']['severities']]
    for name, ds in splits:
        query_bank(model, cfg, name, ds, np.arange(len(ds)), repo, pfull, ck, dev, False, with_cf=(name == 'train'))
        if dev.type == 'cuda': torch.cuda.empty_cache()
    print(f'RESHIFT_PREPARE_PASS target={cfg["experiment"]["target"]} seconds={time.time() - t0:.0f}', flush=True)


if __name__ == '__main__': main()
