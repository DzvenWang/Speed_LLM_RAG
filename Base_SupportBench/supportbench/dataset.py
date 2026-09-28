from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .sensing import materialize_mask
from .utils import read_json, resolve_rel


@dataclass
class ForecastSample:
    subgraph: str
    split: str
    window_start: int
    view_spec: dict[str, Any]
    history_full: np.ndarray
    future_target: np.ndarray
    raw_valid_history: np.ndarray
    raw_valid_future: np.ndarray
    artificial_mask: np.ndarray
    observation_mask: np.ndarray

    @property
    def view_id(self) -> str:
        return str(self.view_spec['view_id'])

    @property
    def view_group_id(self) -> str:
        return str(self.view_spec['view_group_id'])


class SupportBench:
    def __init__(self, subgraph: str, root: str | Path | None = None, config: str | Path | None = None):
        self.cfg = load_config(config)
        self.root = Path(root or self.cfg['project']['benchmark_root']).resolve()
        self.sg_dir = self.root / subgraph
        if not self.sg_dir.exists():
            raise FileNotFoundError(self.sg_dir)
        self.manifest = read_json(self.sg_dir / 'manifest.json')
        self.traffic = np.load(self.sg_dir / 'core' / 'traffic_2024.npy', mmap_mode='r')
        self.valid = np.load(self.sg_dir / 'core' / 'raw_valid_mask.npy', mmap_mode='r')
        self.nodes = pd.read_csv(self.sg_dir / 'core' / 'nodes.csv')
        self.adj = np.load(self.sg_dir / 'graph' / 'adjacency.npy', mmap_mode='r')
        _a = np.asarray(self.adj)
        self.graph_degree = ((_a != 0).sum(axis=0) + (_a != 0).sum(axis=1)).astype(np.int32)
        self.channel_names = list(self.manifest['channels'])
        self.target_name = str(self.manifest['primary_channel']); self.target_idx = self.channel_names.index(self.target_name); self.speed_idx = self.target_idx
        self.H = int(self.manifest['history_len']); self.P = int(self.manifest['horizon']); self.N = int(self.traffic.shape[1])
        self.stats = read_json(self.sg_dir / 'core' / 'normalization.json')
        self.split_manifest = read_json(self.sg_dir / 'splits' / 'split_manifest.json')
        self.views = pd.read_csv(self.sg_dir / 'views' / 'view_manifest.csv')
        if self.views['view_id'].duplicated().any():
            raise ValueError(f'Duplicate view_id detected in {self.sg_dir / "views" / "view_manifest.csv"}')
        self._views_by_id = self.views.set_index('view_id', drop=False)
        directed = np.load(self.sg_dir / 'graph' / 'directed_neighbors.npz')
        self.upstream_ids = directed['upstream_ids'].astype(np.int64)
        self.upstream_weights = directed['upstream_weights'].astype(np.float32)
        self.downstream_ids = directed['downstream_ids'].astype(np.int64)
        self.downstream_weights = directed['downstream_weights'].astype(np.float32)

    def split_range(self, split: str) -> dict:
        return self.split_manifest['splits'][split]

    def window_starts(self, split: str, budget: str = 'full') -> np.ndarray:
        key = f'{split}_{budget}' if budget != 'full' else f'{split}_full'
        rel = self.split_manifest['window_files'][key]
        return np.load(resolve_rel(self.sg_dir, rel)).astype(np.int64)

    def get_view_spec(self, view_id: str) -> dict:
        try:
            row = self._views_by_id.loc[str(view_id)]
        except KeyError as e:
            raise KeyError(view_id) from e
        if isinstance(row, pd.DataFrame):
            if len(row) != 1: raise KeyError(f'duplicate view_id: {view_id}')
            row = row.iloc[0]
        return row.to_dict()

    def views_for(self, split: str, family: str | None = None, severity: float | None = None, distribution: str | None = None) -> pd.DataFrame:
        df = self.views[self.views['split'] == split]
        if family is not None: df = df[df['family'] == family]
        if severity is not None: df = df[np.isclose(df['severity'].astype(float), float(severity))]
        if distribution is not None: df = df[df['distribution'] == distribution]
        return df.reset_index(drop=True)

    def sample(self, view_spec: dict | str) -> ForecastSample:
        spec = self.get_view_spec(view_spec) if isinstance(view_spec, str) else dict(view_spec)
        s = int(spec['window_start']); hs, he, fe = s, s+self.H, s+self.H+self.P
        hist = np.asarray(self.traffic[hs:he], dtype=np.float32)
        future = np.asarray(self.traffic[he:fe, :, self.target_idx], dtype=np.float32)
        valid_hist = np.asarray(self.valid[hs:he], dtype=bool)
        valid_future = np.asarray(self.valid[he:fe, :, self.target_idx], dtype=bool)
        art = materialize_mask(spec, self.H, self.N, self.sg_dir, valid_hist[...,self.target_idx])
        obs = valid_hist & (~art[...,None])
        return ForecastSample(self.sg_dir.name, str(spec['split']), s, spec, hist, future, valid_hist, valid_future, art, obs)

    def normalize(self, x: np.ndarray, channel_indices: list[int] | None = None) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32).copy()
        if channel_indices is None:
            channel_indices = list(range(len(self.channel_names)))
        for local_j, global_j in enumerate(channel_indices):
            st = self.stats['channels'][self.channel_names[global_j]]
            y[...,local_j] = (y[...,local_j] - float(st['mean'])) / max(float(st['std']), 1e-6)
        return y

    def model_view(self, sample: ForecastSample, target_only: bool = True, normalized: bool = True, fill_value: float = 0.0, **legacy) -> dict[str,np.ndarray]:
        if 'speed_only' in legacy: target_only = bool(legacy['speed_only'])
        if target_only:
            full = sample.history_full[..., self.target_idx:self.target_idx+1]; obs = sample.observation_mask[..., self.target_idx:self.target_idx+1]; idx=[self.target_idx]
        else:
            full = sample.history_full; obs = sample.observation_mask; idx=list(range(len(self.channel_names)))
        x = self.normalize(full, idx) if normalized else full.copy()
        x = np.where(obs, x, np.float32(fill_value)).astype(np.float32)
        return {
            'history': x,
            'observation_mask': obs.astype(np.float32),
            'artificial_mask': sample.artificial_mask.astype(np.float32),
            'future_target': sample.future_target.astype(np.float32),
            'future_speed': sample.future_target.astype(np.float32),  # legacy alias; use future_target
            'future_valid_mask': sample.raw_valid_future.astype(np.float32),
        }
