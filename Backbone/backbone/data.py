from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resolve_time_features(bench):
    T = int(bench.traffic.shape[0])
    steps_per_day = int(round(24 * 60 / bench.cfg['forecast']['freq_minutes']))
    p = bench.sg_dir / 'core' / 'time_index.csv'
    if p.exists():
        df = pd.read_csv(p)
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().mean() > 0.95 and len(parsed) >= T:
                dow = parsed.dt.dayofweek.to_numpy(np.int16)[:T]
                tod = ((parsed.dt.hour * 60 + parsed.dt.minute) // int(bench.cfg['forecast']['freq_minutes'])).to_numpy(np.int16)[:T]
                return tod, dow, 'core/time_index.csv'
    idx = np.arange(T, dtype=np.int64)
    return (idx % steps_per_day).astype(np.int16), ((idx // steps_per_day) % 7).astype(np.int16), 'index_modulo_fallback'


def load_protocol_views(cfg, backbone_stage: str):
    from shared_protocol.consumer import load_frozen_stage
    pstage = cfg['protocol']['stage_map'][backbone_stage]
    manifest, views = load_frozen_stage(
        cfg['project']['protocol_artifact_root'],
        cfg['experiment']['subgraph'],
        cfg['protocol']['name'],
        pstage,
    )
    return pstage, manifest, views


def materialize_split(bench, df, outdir, split, tod_all, dow_all, fill_value=0.0):
    H, P, N = bench.H, bench.P, bench.N
    M = len(df)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    xs = np.lib.format.open_memmap(outdir / f'{split}_x_speed_norm.npy', mode='w+', dtype=np.float32, shape=(M, H, N))
    yr = np.lib.format.open_memmap(outdir / f'{split}_y_raw.npy', mode='w+', dtype=np.float32, shape=(M, P, N))
    yn = np.lib.format.open_memmap(outdir / f'{split}_y_norm.npy', mode='w+', dtype=np.float32, shape=(M, P, N))
    fv = np.lib.format.open_memmap(outdir / f'{split}_future_valid.npy', mode='w+', dtype=np.uint8, shape=(M, P, N))
    dm = np.lib.format.open_memmap(outdir / f'{split}_direct_mask.npy', mode='w+', dtype=np.uint8, shape=(M, N))
    sm = np.lib.format.open_memmap(outdir / f'{split}_support_mask.npy', mode='w+', dtype=np.uint8, shape=(M, N))
    th = np.lib.format.open_memmap(outdir / f'{split}_tod_hist.npy', mode='w+', dtype=np.int16, shape=(M, H))
    dh = np.lib.format.open_memmap(outdir / f'{split}_dow_hist.npy', mode='w+', dtype=np.int8, shape=(M, H))
    tf = np.lib.format.open_memmap(outdir / f'{split}_tod_future.npy', mode='w+', dtype=np.int16, shape=(M, P))
    dfu = np.lib.format.open_memmap(outdir / f'{split}_dow_future.npy', mode='w+', dtype=np.int8, shape=(M, P))
    speed_name = bench.manifest['primary_channel']
    mean = float(bench.stats['channels'][speed_name]['mean'])
    std = max(float(bench.stats['channels'][speed_name]['std']), 1e-6)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        s = bench.sample(str(row.view_id))
        mv = bench.model_view(s, speed_only=True, normalized=True, fill_value=float(fill_value))
        xs[i] = mv['history'][..., 0]
        yr[i] = s.future_speed
        yn[i] = (s.future_speed - mean) / std
        fv[i] = s.raw_valid_future.astype(np.uint8)
        dm[i] = s.artificial_mask.any(axis=0).astype(np.uint8)
        sm[i] = np.asarray(bench.predictive_context_metrics(s)['support_dependent_target_mask'], np.uint8)
        st = int(s.window_start)
        th[i] = tod_all[st:st + H]; dh[i] = dow_all[st:st + H]
        tf[i] = tod_all[st + H:st + H + P]; dfu[i] = dow_all[st + H:st + H + P]
        rows.append({
            'protocol_view_id': getattr(row, 'protocol_view_id', ''),
            'view_id': s.view_id, 'view_group_id': s.view_group_id,
            'window_start': st, 'family': str(s.view_spec['family']),
            'severity': float(s.view_spec['severity']),
            'distribution': str(s.view_spec.get('distribution', 'id')),
        })
        if (i + 1) % 500 == 0:
            print(f'materialize {split}: {i+1}/{M}', flush=True)
    for a in [xs, yr, yn, fv, dm, sm, th, dh, tf, dfu]: a.flush()
    pd.DataFrame(rows).to_csv(outdir / f'{split}_manifest.csv', index=False)


class CachedSplit(Dataset):
    def __init__(self, root, split):
        r = Path(root); self.split = split
        self.x = np.load(r / f'{split}_x_speed_norm.npy', mmap_mode='r')
        self.y_raw = np.load(r / f'{split}_y_raw.npy', mmap_mode='r')
        self.y_norm = np.load(r / f'{split}_y_norm.npy', mmap_mode='r')
        self.valid = np.load(r / f'{split}_future_valid.npy', mmap_mode='r')
        self.direct = np.load(r / f'{split}_direct_mask.npy', mmap_mode='r')
        self.support = np.load(r / f'{split}_support_mask.npy', mmap_mode='r')
        self.tod_h = np.load(r / f'{split}_tod_hist.npy', mmap_mode='r')
        self.dow_h = np.load(r / f'{split}_dow_hist.npy', mmap_mode='r')
        self.tod_f = np.load(r / f'{split}_tod_future.npy', mmap_mode='r')
        self.dow_f = np.load(r / f'{split}_dow_future.npy', mmap_mode='r')
        self.meta = pd.read_csv(r / f'{split}_manifest.csv')

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        return {
            'x': torch.from_numpy(np.asarray(self.x[i]).copy()),
            'y_raw': torch.from_numpy(np.asarray(self.y_raw[i]).copy()),
            'y_norm': torch.from_numpy(np.asarray(self.y_norm[i]).copy()),
            'future_valid': torch.from_numpy(np.asarray(self.valid[i], dtype=np.bool_).copy()),
            'direct_mask': torch.from_numpy(np.asarray(self.direct[i], dtype=np.bool_).copy()),
            'support_mask': torch.from_numpy(np.asarray(self.support[i], dtype=np.bool_).copy()),
            'tod_hist': torch.from_numpy(np.asarray(self.tod_h[i], dtype=np.int64).copy()),
            'dow_hist': torch.from_numpy(np.asarray(self.dow_h[i], dtype=np.int64).copy()),
            'tod_future': torch.from_numpy(np.asarray(self.tod_f[i], dtype=np.int64).copy()),
            'dow_future': torch.from_numpy(np.asarray(self.dow_f[i], dtype=np.int64).copy()),
            'index': int(i),
        }
