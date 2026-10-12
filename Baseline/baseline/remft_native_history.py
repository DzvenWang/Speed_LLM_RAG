from __future__ import annotations

"""Native-context input materialization for ReMFT.

The official ReMFT notebook uses seq_len=96 and pred_len=12.  The benchmark's
shared forecasting query stores the last H=12 incomplete steps.  For ReMFT only,
we extend that causal input backwards by 84 steps while preserving the benchmark
query exactly on its last 12 steps.

The extra prefix is *not* complete ground truth: the same view family/severity is
materialized over the prefix, natural invalid cells remain unavailable, and only
observed prefix cells are exposed.  This preserves method-specific context while
respecting incomplete-sensing information availability.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASELINE_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
COMMON = BASELINE_ROOT.parent / 'Base_SupportBench'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from supportbench.sensing import materialize_mask


@dataclass(frozen=True)
class ReMFTNativeSource:
    target: np.ndarray
    valid: np.ndarray
    sg_dir: Path
    view_specs: pd.DataFrame
    native_history: int
    benchmark_history: int
    nodes: int
    mean: float
    std: float

    @property
    def prefix_len(self) -> int:
        return self.native_history - self.benchmark_history

    def _spec(self, view_id: str) -> dict:
        try:
            row = self.view_specs.loc[str(view_id)]
        except KeyError as exc:
            raise KeyError(f'ReMFT view_id not found in SupportBench view manifest: {view_id}') from exc
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise RuntimeError(f'ReMFT duplicate view_id in SupportBench: {view_id}')
            row = row.iloc[0]
        spec = row.to_dict()
        # Convert numpy scalars to ordinary Python values because SupportBench
        # view helpers expect JSON-like dictionaries.
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in spec.items()}

    def build(
        self,
        *,
        window_start: int,
        view_id: str,
        tail_x_obs_norm: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        st = int(window_start)
        tail = np.asarray(tail_x_obs_norm, dtype=np.float32)
        if tail.shape != (self.benchmark_history, self.nodes):
            raise ValueError(
                f'ReMFT benchmark-tail shape {tail.shape} != '
                f'{(self.benchmark_history, self.nodes)}'
            )
        ps = st - self.prefix_len
        if ps < 0:
            raise IndexError(
                f'ReMFT native H{self.native_history} prefix unavailable for window_start={st}; '
                f'prefix_start={ps}'
            )
        raw = np.asarray(self.target[ps:st], dtype=np.float32)
        valid = np.asarray(self.valid[ps:st], dtype=bool)
        if raw.shape != (self.prefix_len, self.nodes) or valid.shape != raw.shape:
            raise RuntimeError(
                f'ReMFT native prefix truncated: raw={raw.shape} valid={valid.shape} '
                f'expected={(self.prefix_len, self.nodes)}'
            )

        spec = self._spec(view_id)
        # Re-materialize only the extra prefix.  The cache's official H12 tail is
        # appended unchanged, so every method still sees exactly the same current
        # benchmark query.  For structured families, using the same view seed/layout
        # preserves the affected node set; for entry-random it preserves severity.
        artificial = materialize_mask(
            spec,
            self.prefix_len,
            self.nodes,
            self.sg_dir,
            raw_valid_speed=valid,
        )
        observed = valid & (~artificial)
        prefix_norm = (raw - np.float32(self.mean)) / np.float32(self.std)
        prefix_obs_norm = np.where(observed, prefix_norm, np.float32(0.0)).astype(np.float32)
        x96 = np.concatenate([prefix_obs_norm, tail], axis=0)
        if x96.shape != (self.native_history, self.nodes):
            raise RuntimeError(f'ReMFT native input shape mismatch: {x96.shape}')
        meta = {
            'prefix_start': ps,
            'prefix_end_exclusive': st,
            'prefix_observed_fraction': float(observed.mean()),
            'prefix_artificial_missing_fraction': float(artificial.mean()),
            'policy': 'same_view_prefix_rematerialization_plus_exact_benchmark_tail',
        }
        return x96, meta


def _target_channel(arr: np.ndarray, idx: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, idx]
    raise RuntimeError(f'Unsupported ReMFT canonical array shape={arr.shape}')


def resolve_remft_native_source(
    cfg: dict,
    *,
    mean: float,
    std: float,
) -> ReMFTNativeSource:
    c = cfg['runners']['ReMFT']
    native_history = int(c.get('native_seq_len', 96))
    benchmark_history = int(cfg['experiment']['history'])
    if native_history != 96:
        raise RuntimeError(
            f'ReMFT corrected main-track implementation requires native_seq_len=96, got {native_history}'
        )
    if native_history <= benchmark_history:
        raise RuntimeError('ReMFT native history must be longer than benchmark H12 query')

    sg_dir = Path(cfg['project']['benchmark_root']) / cfg['experiment']['subgraph']
    manifest_path = sg_dir / 'manifest.json'
    traffic_path = sg_dir / 'core' / 'traffic_2024.npy'
    valid_path = sg_dir / 'core' / 'raw_valid_mask.npy'
    views_path = sg_dir / 'views' / 'view_manifest.csv'
    for p in (manifest_path, traffic_path, valid_path, views_path):
        if not p.exists():
            raise FileNotFoundError(p)

    manifest = json.loads(manifest_path.read_text())
    channels = list(manifest.get('channels', []))
    primary = str(manifest.get('primary_channel', cfg['experiment']['target']))
    if channels:
        if primary not in channels:
            raise RuntimeError(f'ReMFT primary_channel={primary} not in benchmark channels={channels}')
        target_idx = channels.index(primary)
    else:
        target_idx = 0

    traffic = np.load(traffic_path, mmap_mode='r')
    valid_all = np.load(valid_path, mmap_mode='r')
    target = _target_channel(traffic, target_idx)
    valid = _target_channel(valid_all, target_idx).astype(bool)
    if target.shape != valid.shape or target.ndim != 2:
        raise RuntimeError(
            f'ReMFT canonical target/valid shape mismatch target={target.shape} valid={valid.shape}'
        )
    views = pd.read_csv(views_path)
    if 'view_id' not in views.columns:
        raise RuntimeError(f'ReMFT SupportBench view manifest lacks view_id: {views_path}')
    views = views.set_index('view_id', drop=False)
    return ReMFTNativeSource(
        target=target,
        valid=valid,
        sg_dir=sg_dir,
        view_specs=views,
        native_history=native_history,
        benchmark_history=benchmark_history,
        nodes=int(target.shape[1]),
        mean=float(mean),
        std=max(float(std), 1e-6),
    )
