from __future__ import annotations

"""Causal multi-day tensor construction for the LRTC-TMCP baseline.

LRTC-TMCP is a tensor-completion method whose traffic use case relies on a
location x day x within-day-interval tensor.  A single-day ``[N,1,H]`` tensor
collapses the day mode and is therefore not a faithful compatibility adapter.

This module resolves the canonical raw-target/raw-valid timeline from
SupportBench using only currently observed benchmark cells for source auditing,
then constructs a strictly causal same-slot multi-day tensor for each query:

    [N, historical_days + 1, H]

All historical slices are at least one full day before the current query.  The
final/current slice contains only the incomplete benchmark H-step history.
No future target value is ever read.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def _array_channel(a: np.ndarray, channel: int | None) -> np.ndarray:
    if a.ndim == 2:
        if channel is not None:
            raise ValueError('2-D canonical array cannot have a channel index')
        return a
    if a.ndim == 3 and channel is not None:
        return a[:, :, channel]
    raise ValueError(f'Unsupported canonical array shape={a.shape}, channel={channel}')


def _sample_is_binary(a: np.ndarray) -> bool:
    flat = np.asarray(a).reshape(-1)
    if flat.size > 20000:
        flat = flat[:20000]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return False
    return bool(np.all((flat == 0) | (flat == 1)))


def _probe_indices(n: int) -> list[int]:
    if n <= 0:
        raise ValueError('LRTC-TMCP source resolver requires at least one query')
    return sorted(set([0, n // 2, n - 1]))


def _sha256_ints(values: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclass(frozen=True)
class LRTCCanonicalHistory:
    target_array: np.ndarray
    target_channel: int | None
    valid_array: np.ndarray
    valid_channel: int | None
    target_path: Path
    valid_path: Path
    history: int
    nodes: int
    historical_days: int
    day_steps: int
    observed_match_mae: float
    raw_valid_agreement: float

    @property
    def context_days(self) -> int:
        return self.historical_days + 1

    def _target_slice(self, start: int) -> np.ndarray:
        return np.asarray(
            _array_channel(self.target_array, self.target_channel)[start:start + self.history],
            dtype=np.float32,
        )

    def _valid_slice(self, start: int) -> np.ndarray:
        return np.asarray(
            _array_channel(self.valid_array, self.valid_channel)[start:start + self.history],
        ).astype(bool)

    def starts_for_query(self, window_start: int) -> list[int]:
        st = int(window_start)
        historical = [st - self.day_steps * k for k in range(self.historical_days, 0, -1)]
        if not historical or historical[0] < 0:
            raise RuntimeError(
                f'LRTC-TMCP requires {self.historical_days} complete causal historical days, '
                f'but window_start={st} gives earliest_start={historical[0] if historical else None}'
            )
        # Historical H-step slices must finish no later than the current query start.
        if any(s + self.history > st for s in historical):
            raise RuntimeError(
                f'LRTC-TMCP causal history violation: starts={historical[-3:]}, current_start={st}'
            )
        return historical + [st]

    def build_tensor(
        self,
        window_start: int,
        current_raw: np.ndarray,
        current_observed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Build one [N,D,H] sparse tensor + observed mask.

        The previous ``historical_days`` slices come from the canonical sensor
        timeline. The final slice is the current incomplete benchmark H-step
        history, so artificially hidden current cells remain unavailable.
        """
        current_raw = np.asarray(current_raw, dtype=np.float32)
        current_observed = np.asarray(current_observed).astype(bool)
        if current_raw.shape != (self.history, self.nodes):
            raise ValueError(
                f'LRTC current query shape {current_raw.shape} != {(self.history, self.nodes)}'
            )
        if current_observed.shape != current_raw.shape:
            raise ValueError('LRTC current observed-mask shape mismatch')

        starts = self.starts_for_query(int(window_start))
        D = self.context_days
        values = np.zeros((self.nodes, D, self.history), dtype=np.float32)
        mask = np.zeros((self.nodes, D, self.history), dtype=bool)

        for d, hs in enumerate(starts[:-1]):
            target = self._target_slice(hs)
            valid = self._valid_slice(hs)
            if target.shape != (self.history, self.nodes) or valid.shape != target.shape:
                raise RuntimeError(f'LRTC canonical context truncated at historical start={hs}')
            good = valid & np.isfinite(target)
            values[:, d, :] = np.where(good, target, 0.0).T
            mask[:, d, :] = good.T

        current_good = current_observed & np.isfinite(current_raw)
        values[:, -1, :] = np.where(current_good, current_raw, 0.0).T
        mask[:, -1, :] = current_good.T
        return values, mask, starts

    def split_index_hashes(self, window_starts: np.ndarray) -> dict[str, str]:
        starts = np.asarray(window_starts, dtype=np.int64)
        offsets = -self.day_steps * np.arange(self.historical_days, 0, -1, dtype=np.int64)
        hist = starts[:, None] + offsets[None, :]
        all_starts = np.concatenate([hist, starts[:, None]], axis=1)
        return {
            'query_window_start_sha256': _sha256_ints(starts),
            'causal_context_start_sha256': _sha256_ints(all_starts),
        }


def _iter_core_channels(core_dir: Path, *, nodes: int, min_time: int):
    for path in sorted(core_dir.glob('*.npy')):
        try:
            arr = np.load(path, mmap_mode='r')
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] >= min_time and arr.shape[1] == nodes:
            yield path, arr, None
        elif arr.ndim == 3 and arr.shape[0] >= min_time and arr.shape[1] == nodes:
            for ch in range(arr.shape[2]):
                yield path, arr, ch


def resolve_lrtc_canonical_history_source(
    cfg: dict,
    cache: Path,
    split: str,
    *,
    mean: float,
    std: float,
) -> LRTCCanonicalHistory:
    """Resolve canonical raw target/raw-valid arrays without hidden-value leakage."""
    c = cfg['runners']['LRTC-TMCP']
    H = int(cfg['experiment']['history'])
    historical_days = int(c['historical_days'])
    day_steps = int(c['day_steps'])
    if H <= 0 or historical_days < 2 or day_steps <= H:
        raise ValueError(
            f'Invalid LRTC geometry H={H}, historical_days={historical_days}, day_steps={day_steps}'
        )

    cache = Path(cache)
    meta = pd.read_csv(cache / f'{split}_manifest.csv')
    x_obs = np.load(cache / f'{split}_x_obs_norm.npy', mmap_mode='r')
    observed = np.load(cache / f'{split}_history_observed.npy', mmap_mode='r')
    artificial = np.load(cache / f'{split}_artificial_missing.npy', mmap_mode='r')
    if x_obs.ndim != 3:
        raise RuntimeError(f'LRTC expected {split}_x_obs_norm [M,H,N], got {x_obs.shape}')
    M, Hx, N = map(int, x_obs.shape)
    if Hx != H or len(meta) != M:
        raise RuntimeError('LRTC cache geometry mismatch')

    min_start = int(meta.window_start.min())
    if min_start - historical_days * day_steps < 0:
        raise RuntimeError(
            f'LRTC split={split} lacks {historical_days} causal historical days for earliest query: '
            f'min_window_start={min_start}, required_backstep={historical_days * day_steps}'
        )
    max_time = int(meta.window_start.max()) + H

    sg_dir = Path(cfg['project']['benchmark_root']) / cfg['experiment']['subgraph']
    core_dir = sg_dir / 'core'
    if not core_dir.exists():
        raise FileNotFoundError(f'LRTC canonical SupportBench core directory not found: {core_dir}')

    probes = _probe_indices(M)
    target_scores: list[tuple[float, Path, np.ndarray, int | None]] = []
    valid_scores: list[tuple[float, Path, np.ndarray, int | None]] = []
    for path, arr, ch in _iter_core_channels(core_dir, nodes=N, min_time=max_time):
        view = _array_channel(arr, ch)
        p0 = int(meta.iloc[probes[0]].window_start)
        sample = np.asarray(view[p0:p0 + H])

        if np.issubdtype(view.dtype, np.number) and not _sample_is_binary(sample):
            errs = []
            for qi in probes:
                st = int(meta.iloc[qi].window_start)
                obs = np.asarray(observed[qi]).astype(bool)
                qraw = np.asarray(x_obs[qi], np.float32) * np.float32(std) + np.float32(mean)
                cand = np.asarray(view[st:st + H], np.float32)
                m = obs & np.isfinite(qraw) & np.isfinite(cand)
                if np.any(m):
                    errs.append(float(np.mean(np.abs(cand[m] - qraw[m]))))
            if errs:
                target_scores.append((float(np.mean(errs)), path, arr, ch))

        if _sample_is_binary(sample):
            agreements = []
            for qi in probes:
                st = int(meta.iloc[qi].window_start)
                # raw-valid contains both currently observed and artificially hidden
                # cells, but this is mask identity only and exposes no hidden values.
                expected = (
                    np.asarray(observed[qi]).astype(bool)
                    | np.asarray(artificial[qi]).astype(bool)
                )
                cand = np.asarray(view[st:st + H]).astype(bool)
                if cand.shape == expected.shape:
                    agreements.append(float(np.mean(cand == expected)))
            if agreements:
                valid_scores.append((float(np.mean(agreements)), path, arr, ch))

    if not target_scores:
        raise RuntimeError(f'LRTC could not find canonical target candidate under {core_dir}')
    target_scores.sort(key=lambda z: z[0])
    target_mae, target_path, target_arr, target_ch = target_scores[0]
    target_tol = float(c.get('canonical_target_match_mae_max', 0.05))
    if target_mae > target_tol:
        raise RuntimeError(
            f'LRTC canonical target audit failed: best={target_path} channel={target_ch} '
            f'observed-cell MAE={target_mae:.6f} > {target_tol}'
        )

    if not valid_scores:
        raise RuntimeError(f'LRTC could not find canonical raw-valid mask candidate under {core_dir}')
    valid_scores.sort(key=lambda z: z[0], reverse=True)
    valid_agree, valid_path, valid_arr, valid_ch = valid_scores[0]
    valid_min = float(c.get('canonical_valid_agreement_min', 0.9999))
    if valid_agree < valid_min:
        raise RuntimeError(
            f'LRTC canonical valid-mask audit failed: best={valid_path} channel={valid_ch} '
            f'agreement={valid_agree:.6f} < {valid_min}'
        )

    source = LRTCCanonicalHistory(
        target_array=target_arr,
        target_channel=target_ch,
        valid_array=valid_arr,
        valid_channel=valid_ch,
        target_path=target_path,
        valid_path=valid_path,
        history=H,
        nodes=N,
        historical_days=historical_days,
        day_steps=day_steps,
        observed_match_mae=float(target_mae),
        raw_valid_agreement=float(valid_agree),
    )
    print('LRTC-TMCP CANONICAL HISTORY SOURCE', {
        'target_path': str(target_path), 'target_channel': target_ch,
        'observed_match_mae': target_mae,
        'valid_path': str(valid_path), 'valid_channel': valid_ch,
        'raw_valid_agreement': valid_agree,
        'historical_days': historical_days, 'context_days': source.context_days,
        'day_steps': day_steps, 'history_intervals': H,
        'tensor_layout': 'nodes_x_causal_days_x_query_slots',
        'causality': 'historical slices strictly earlier; current hidden and every P12 future unavailable',
    }, flush=True)
    return source
