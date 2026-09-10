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
    future_speed: np.ndarray
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
        self.speed_idx = self.channel_names.index(self.manifest['primary_channel'])
        self.H = int(self.manifest['history_len']); self.P = int(self.manifest['horizon']); self.N = int(self.traffic.shape[1])
        self.stats = read_json(self.sg_dir / 'core' / 'normalization.json')
        self.split_manifest = read_json(self.sg_dir / 'splits' / 'split_manifest.json')
        self.views = pd.read_csv(self.sg_dir / 'views' / 'view_manifest.csv')
        if self.views['view_id'].duplicated().any():
            raise ValueError(f'Duplicate view_id detected in {self.sg_dir / "views" / "view_manifest.csv"}')
        self._views_by_id = self.views.set_index('view_id', drop=False)
        self.memory_manifest = read_json(self.sg_dir / 'memory' / 'memory_manifest.json')
        self.memory_starts = np.load(self.sg_dir / 'memory' / 'episode_window_starts_full.npy')
        self.memory_starts_development = np.load(self.sg_dir / 'memory' / 'episode_window_starts_development.npy')
        dep=np.load(self.sg_dir / 'graph' / 'predictive_dependency.npz')
        self.dependency_ids=dep['ids'].astype(np.int64); self.dependency_weights=dep['weights'].astype(np.float32)
        self.dependency_lags=dep['lags'].astype(np.int16)
        self.dependency_lag_step_minutes=int(dep['lag_step_minutes']) if 'lag_step_minutes' in dep.files else int(self.cfg['forecast']['freq_minutes'])
        self.dependency_lag_minutes=(dep['lag_minutes'].astype(np.int16) if 'lag_minutes' in dep.files else np.where(self.dependency_lags>=0,self.dependency_lags*self.dependency_lag_step_minutes,-1).astype(np.int16))
        self.dependency_corr=dep['corr'].astype(np.float32); self.dependency_relation=dep['relation'].astype(np.int8)
        self.predictive_source_centrality=dep['source_centrality'].astype(np.float32)
        patches = np.load(self.sg_dir / 'memory' / 'local_patches.npz')
        self.patch_ids = patches['ids']; self.patch_valid = patches['valid'].astype(bool)
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
        future = np.asarray(self.traffic[he:fe, :, self.speed_idx], dtype=np.float32)
        valid_hist = np.asarray(self.valid[hs:he], dtype=bool)
        valid_future = np.asarray(self.valid[he:fe, :, self.speed_idx], dtype=bool)
        art = materialize_mask(spec, self.H, self.N, self.sg_dir, valid_hist[...,self.speed_idx])
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

    def model_view(self, sample: ForecastSample, speed_only: bool = True, normalized: bool = True, fill_value: float = 0.0) -> dict[str,np.ndarray]:
        if speed_only:
            full = sample.history_full[..., self.speed_idx:self.speed_idx+1]; obs = sample.observation_mask[..., self.speed_idx:self.speed_idx+1]; idx=[self.speed_idx]
        else:
            full = sample.history_full; obs = sample.observation_mask; idx=list(range(len(self.channel_names)))
        x = self.normalize(full, idx) if normalized else full.copy()
        x = np.where(obs, x, np.float32(fill_value)).astype(np.float32)
        return {
            'history': x,
            'observation_mask': obs.astype(np.float32),
            'artificial_mask': sample.artificial_mask.astype(np.float32),
            'future_speed': sample.future_speed.astype(np.float32),
            'future_valid_mask': sample.raw_valid_future.astype(np.float32),
        }

    def memory_starts_for_budget(self, budget: str='full') -> np.ndarray:
        if str(budget)=='development': return np.asarray(self.memory_starts_development,np.int64)
        if str(budget)=='full': return np.asarray(self.memory_starts,np.int64)
        raise KeyError(f'Unknown memory budget: {budget}')

    def memory_episode_ids_for_budget(self, budget: str='full') -> np.ndarray:
        if str(budget)=='full': return np.arange(len(self.memory_starts),dtype=np.int64)
        if str(budget)!='development': raise KeyError(f'Unknown memory budget: {budget}')
        pos={int(s):i for i,s in enumerate(np.asarray(self.memory_starts,np.int64))}
        return np.asarray([pos[int(s)] for s in np.asarray(self.memory_starts_development,np.int64) if int(s) in pos],dtype=np.int64)

    def memory_episode_from_start(self, window_start: int) -> dict[str,np.ndarray | int]:
        s=int(window_start); he=s+self.H; fe=he+self.P
        return {'window_start':s,'history_speed':np.asarray(self.traffic[s:he,:,self.speed_idx],dtype=np.float32),'future_speed':np.asarray(self.traffic[he:fe,:,self.speed_idx],dtype=np.float32)}

    def predictive_context_metrics(self, sample: ForecastSample) -> dict[str,np.ndarray | float]:
        """Return Train-defined predictive-context diagnostics with explicit semantics.

        Formal paper metric:
            network_mean_context_loss = mean(per_target_context_loss)

        Backward compatibility:
            aggregate_context_loss keeps the v1.2.2 q-quantile semantics.
            New code must not use it as Chapter-3 C_q^loss.
        """
        from .dependency import dependency_context_loss, summarize_dependency_context_loss
        loss=dependency_context_loss(sample.artificial_mask,self.dependency_ids,self.dependency_weights)
        obs=np.asarray(sample.observation_mask[...,self.speed_idx],bool).mean(axis=0)
        dcfg=self.cfg['dependency']; support=(loss>=float(dcfg['support_dependent_missing_threshold'])) & (obs>=float(dcfg['support_dependent_target_observed_min']))
        q=float(dcfg.get('context_loss_quantile',0.75))
        summary=summarize_dependency_context_loss(loss,quantile=q)
        legacy=float(summary['context_loss_quantile_value'])
        return {
            'per_target_context_loss':loss.astype(np.float32),
            'support_dependent_target_mask':support.astype(bool),
            'network_mean_context_loss':float(summary['network_mean_context_loss']),
            'predictive_context_loss':float(summary['network_mean_context_loss']),
            'context_loss_quantile':float(summary['context_loss_quantile']),
            'context_loss_quantile_value':legacy,
            'aggregate_context_loss':legacy,  # legacy v1.2.2 alias
            'legacy_aggregate_context_loss':legacy,
            'support_dependent_fraction':float(support.mean()) if len(support) else 0.0,
        }

    def memory_episode(self, episode_id: int) -> dict[str,np.ndarray | int]:
        s = int(self.memory_starts[int(episode_id)]); he = s+self.H; fe=he+self.P
        return {
            'episode_id': int(episode_id), 'window_start': s,
            'history_speed': np.asarray(self.traffic[s:he,:,self.speed_idx], dtype=np.float32),
            'future_speed': np.asarray(self.traffic[he:fe,:,self.speed_idx], dtype=np.float32),
        }

    def local_patch_nodes(self, target_node: int) -> np.ndarray:
        ids = self.patch_ids[int(target_node)]
        valid = self.patch_valid[int(target_node)]
        return ids[valid].astype(np.int64)
