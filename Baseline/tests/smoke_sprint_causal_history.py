from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.sprint_causal_history import resolve_canonical_history_source


cfg = yaml.safe_load((ROOT / 'configs/baseline.yaml').read_text())
H = int(cfg['experiment']['history'])
assert H == 12

with tempfile.TemporaryDirectory(prefix='sprint_causal_history_smoke_') as td:
    td = Path(td)
    benchmark_root = td / 'bench'
    subgraph = 'SyntheticNet'
    core = benchmark_root / subgraph / 'core'
    cache = td / 'cache'
    core.mkdir(parents=True)
    cache.mkdir(parents=True)

    # Long enough for the configured previous-day causal context.
    T, N = 12000, 4
    tt = np.arange(T, dtype=np.float32)[:, None]
    nn = np.arange(N, dtype=np.float32)[None, :]
    speed = 55.0 + 0.002 * tt + 0.25 * nn
    traffic = np.stack([
        150.0 + 0.01 * tt + nn,            # flow-like distractor
        0.05 + 0.00001 * tt + 0.001 * nn,  # occupancy-like distractor
        speed,
    ], axis=-1).astype(np.float32)
    valid = np.ones((T, N), dtype=bool)
    valid[1000::97, 2] = False
    np.save(core / 'traffic.npy', traffic)
    np.save(core / 'raw_valid_mask.npy', valid)

    starts = np.asarray([9000, 9500, 10000], dtype=np.int64)
    M = len(starts)
    observed = np.zeros((M, H, N), dtype=bool)
    artificial = np.zeros_like(observed)
    x_obs_norm = np.zeros((M, H, N), dtype=np.float32)
    mean, std = 60.0, 10.0

    for i, st in enumerate(starts):
        raw_valid = valid[st:st + H]
        # Keep a deterministic subset observed; hide the rest artificially.
        obs = raw_valid.copy()
        obs[:, 1] = False
        observed[i] = obs
        artificial[i] = raw_valid & ~obs
        raw = speed[st:st + H]
        norm = (raw - mean) / std
        x_obs_norm[i] = np.where(obs, norm, 0.0)

    np.save(cache / 'val_x_obs_norm.npy', x_obs_norm)
    np.save(cache / 'val_history_observed.npy', observed)
    np.save(cache / 'val_artificial_missing.npy', artificial)
    pd.DataFrame({
        'view_id': [f'v{i}' for i in range(M)],
        'window_start': starts,
    }).to_csv(cache / 'val_manifest.csv', index=False)
    (cache / 'cache_manifest.json').write_text(json.dumps({
        'splits': {'val': {'mean': mean, 'std': std}},
    }))

    cfg2 = dict(cfg)
    cfg2['project'] = dict(cfg['project'])
    cfg2['experiment'] = dict(cfg['experiment'])
    cfg2['runners'] = dict(cfg['runners'])
    cfg2['runners']['SPRINT'] = dict(cfg['runners']['SPRINT'])
    cfg2['project']['benchmark_root'] = str(benchmark_root)
    cfg2['experiment']['subgraph'] = subgraph

    source = resolve_canonical_history_source(cfg2, cache, 'val', mean=mean, std=std)
    assert source.speed_path.name == 'traffic.npy'
    assert source.speed_channel == 2
    assert source.valid_path.name == 'raw_valid_mask.npy'
    D=int(cfg2['runners']['SPRINT']['historical_days'])
    assert source.context_days == D + 1

    qi = 0
    st = int(starts[qi])
    current_raw = x_obs_norm[qi] * std + mean
    context, hist_starts = source.build_context(st, current_raw, observed[qi])
    assert context.shape == (D + 1, H, N)
    assert len(hist_starts) == D
    assert hist_starts[0] == st - D * 288
    assert hist_starts[-1] == st - 288
    assert all(hs + H <= st for hs in hist_starts)

    # Current artificially hidden values remain unavailable to SPRINT.
    assert np.isnan(context[-1][~observed[qi]]).all()
    np.testing.assert_allclose(context[-1][observed[qi]], speed[st:st+H][observed[qi]], atol=1e-5)

    # Historical native-invalid positions are masked rather than silently filled.
    for d, hs in enumerate(hist_starts):
        expected_valid = valid[hs:hs+H]
        assert np.isnan(context[d][~expected_valid]).all()
        np.testing.assert_allclose(context[d][expected_valid], speed[hs:hs+H][expected_valid], atol=1e-6)

print('SPRINT CAUSAL HISTORICAL-DAY SOURCE SMOKE PASS')
