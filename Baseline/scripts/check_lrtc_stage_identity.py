#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.runners.common import config
from baseline.shared_cache import cache_path


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha_ints(a) -> str:
    x = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    return hashlib.sha256(x.tobytes()).hexdigest()


def stage_signature(cfg: dict, stage: str) -> dict:
    cache = cache_path(cfg, stage)
    meta = pd.read_csv(cache / 'val_manifest.csv')
    c = cfg['runners']['LRTC-TMCP']
    days = int(c['historical_days'])
    step = int(c['day_steps'])
    starts = meta.window_start.astype(np.int64).to_numpy()
    offsets = -step * np.arange(days, 0, -1, dtype=np.int64)
    context = np.concatenate([starts[:, None] + offsets[None, :], starts[:, None]], axis=1)
    return {
        'stage': stage,
        'val_rows': int(len(meta)),
        'query_window_start_sha256': sha_ints(starts),
        'causal_context_start_sha256': sha_ints(context),
        'x_obs_file_sha256': sha_file(cache / 'val_x_obs_norm.npy'),
        'history_observed_file_sha256': sha_file(cache / 'val_history_observed.npy'),
        'history_days': days,
        'day_steps': step,
    }


def main():
    cfg = config()
    dev = stage_signature(cfg, 'development')
    formal = stage_signature(cfg, 'formal')
    keys = [
        'query_window_start_sha256',
        'causal_context_start_sha256',
        'x_obs_file_sha256',
        'history_observed_file_sha256',
    ]
    same = {k: dev[k] == formal[k] for k in keys}
    all_same = all(same.values())
    result = {'development': dev, 'formal': formal, 'same': same, 'all_deterministic_inputs_same': all_same}
    print(json.dumps(result, indent=2, sort_keys=True))
    if all_same:
        print('LRTC STAGE IDENTITY: deterministic Development/Formal predictions may legitimately be bit-identical.')
    else:
        print('LRTC STAGE IDENTITY: inputs differ; bit-identical predictions would require further cache/runner audit.')


if __name__ == '__main__':
    main()
