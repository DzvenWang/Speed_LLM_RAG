from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        raise ValueError('SPRINT source resolver requires at least one query')
    return sorted(set([0, n // 2, n - 1]))


@dataclass
class SprintCanonicalHistory:
    """Read-only timeline-causal historical context for specialized SPRINT.

    Historical context may use canonical sensor observations from earlier timestamps,
    including earlier timestamps inside an evaluation period, but it never uses the
    current query's hidden artificial values, P12 future values, later timestamps,
    predictions, or labels. This extra causal context is why SPRINT is reported in a
    specialized comparator track rather than the strict H12 input-matched main table.
    """

    speed_array: np.ndarray
    speed_channel: int | None
    valid_array: np.ndarray
    valid_channel: int | None
    speed_path: Path
    valid_path: Path
    history: int
    nodes: int
    historical_days: int
    day_steps: int

    @property
    def context_days(self) -> int:
        return self.historical_days + 1

    def _speed_slice(self, start: int) -> np.ndarray:
        return np.asarray(
            _array_channel(self.speed_array, self.speed_channel)[start:start + self.history],
            dtype=np.float32,
        )

    def _valid_slice(self, start: int) -> np.ndarray:
        return np.asarray(
            _array_channel(self.valid_array, self.valid_channel)[start:start + self.history],
        ).astype(bool)

    def build_context(
        self,
        window_start: int,
        current_raw: np.ndarray,
        current_observed: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Return [historical_days+1,H,N] with strict timestamp causality."""
        st = int(window_start)
        current_raw = np.asarray(current_raw, np.float32)
        current_observed = np.asarray(current_observed).astype(bool)
        if current_raw.shape != (self.history, self.nodes):
            raise ValueError(f'Current SPRINT query shape {current_raw.shape} != {(self.history, self.nodes)}')
        if current_observed.shape != current_raw.shape:
            raise ValueError('Current SPRINT observed mask shape mismatch')

        starts = [st - self.day_steps * k for k in range(self.historical_days, 0, -1)]
        if not starts or starts[0] < 0:
            raise RuntimeError(
                f'SPRINT requires {self.historical_days} complete causal historical days, '
                f'but window_start={st} gives earliest_start={starts[0] if starts else None}. '
                'Do not shorten context silently; revise the specialized-comparator protocol explicitly.'
            )
        # Strictly earlier than the current query origin. Equality would still be safe
        # for half-open intervals, but using <= makes the contract easy to audit.
        if any(s + self.history > st for s in starts):
            raise RuntimeError(
                f'SPRINT causal history violation: historical starts={starts[-3:]}, current_start={st}'
            )

        context = np.full((self.context_days, self.history, self.nodes), np.nan, np.float32)
        for d, hs in enumerate(starts):
            speed = self._speed_slice(hs)
            valid = self._valid_slice(hs)
            if speed.shape != (self.history, self.nodes) or valid.shape != speed.shape:
                raise RuntimeError(f'SPRINT canonical context truncated at historical start={hs}')
            context[d] = np.where(valid & np.isfinite(speed), speed, np.nan)

        # Current day: only benchmark-observed H12 cells are visible. Artificially
        # hidden current-history cells and every P12 future value remain unavailable.
        context[-1] = np.where(current_observed & np.isfinite(current_raw), current_raw, np.nan)
        return context, starts


def _iter_core_channels(core_dir: Path, *, nodes: int, min_time: int):
    """Yield mmap-backed [T,N] or [T,N,C] canonical candidates and channel ids."""
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


def resolve_canonical_history_source(
    cfg: dict,
    cache: Path,
    split: str,
    *,
    mean: float,
    std: float,
) -> SprintCanonicalHistory:
    """Resolve canonical raw speed + raw-valid mask using the frozen split as audit.

    Candidate speed arrays are scored only on *currently observed* cells, so hidden
    current-query values cannot select the source. Raw-valid candidates are scored
    against frozen `history_observed OR artificial_missing`, which is mask identity
    only and contains no hidden speed values.
    """
    c = cfg['runners']['SPRINT']
    H = int(cfg['experiment']['history'])
    historical_days = int(c['historical_days'])
    day_steps = int(c['day_steps'])
    if H <= 0 or historical_days < 2 or day_steps <= H:
        raise ValueError(
            f'Invalid SPRINT causal-history geometry H={H}, historical_days={historical_days}, day_steps={day_steps}'
        )

    cache = Path(cache)
    meta = pd.read_csv(cache / f'{split}_manifest.csv')
    x_obs = np.load(cache / f'{split}_x_obs_norm.npy', mmap_mode='r')
    observed = np.load(cache / f'{split}_history_observed.npy', mmap_mode='r')
    artificial = np.load(cache / f'{split}_artificial_missing.npy', mmap_mode='r')
    if x_obs.ndim != 3:
        raise RuntimeError(f'SPRINT expected {split}_x_obs_norm [M,H,N], got {x_obs.shape}')
    M, Hx, N = map(int, x_obs.shape)
    if Hx != H or len(meta) != M:
        raise RuntimeError('SPRINT cache geometry mismatch')
    max_time = int(meta.window_start.max()) + H

    sg_dir = Path(cfg['project']['benchmark_root']) / cfg['experiment']['subgraph']
    core_dir = sg_dir / 'core'
    if not core_dir.exists():
        raise FileNotFoundError(f'SPRINT canonical SupportBench core directory not found: {core_dir}')

    probes = _probe_indices(M)
    speed_scores: list[tuple[float, Path, np.ndarray, int | None]] = []
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
                speed_scores.append((float(np.mean(errs)), path, arr, ch))

        if _sample_is_binary(sample):
            agreements = []
            for qi in probes:
                st = int(meta.iloc[qi].window_start)
                expected = (
                    np.asarray(observed[qi]).astype(bool)
                    | np.asarray(artificial[qi]).astype(bool)
                )
                cand = np.asarray(view[st:st + H]).astype(bool)
                if cand.shape == expected.shape:
                    agreements.append(float(np.mean(cand == expected)))
            if agreements:
                valid_scores.append((float(np.mean(agreements)), path, arr, ch))

    if not speed_scores:
        raise RuntimeError(f'SPRINT could not find a canonical speed candidate under {core_dir}')
    speed_scores.sort(key=lambda z: z[0])
    speed_mae, speed_path, speed_arr, speed_ch = speed_scores[0]
    speed_tol = float(c.get('canonical_speed_match_mae_max', 0.05))
    if speed_mae > speed_tol:
        raise RuntimeError(
            f'SPRINT canonical speed source audit failed: best={speed_path} channel={speed_ch} '
            f'observed-cell MAE={speed_mae:.6f} > {speed_tol}'
        )

    if not valid_scores:
        raise RuntimeError(f'SPRINT could not find a canonical raw-valid mask candidate under {core_dir}')
    valid_scores.sort(key=lambda z: z[0], reverse=True)
    valid_agree, valid_path, valid_arr, valid_ch = valid_scores[0]
    valid_min = float(c.get('canonical_valid_agreement_min', 0.9999))
    if valid_agree < valid_min:
        raise RuntimeError(
            f'SPRINT canonical valid-mask audit failed: best={valid_path} channel={valid_ch} '
            f'agreement={valid_agree:.6f} < {valid_min}'
        )

    source = SprintCanonicalHistory(
        speed_array=speed_arr, speed_channel=speed_ch,
        valid_array=valid_arr, valid_channel=valid_ch,
        speed_path=speed_path, valid_path=valid_path,
        history=H, nodes=N, historical_days=historical_days, day_steps=day_steps,
    )
    min_start = int(meta.window_start.min())
    if min_start - historical_days * day_steps < 0:
        raise RuntimeError(
            f'SPRINT split={split} lacks {historical_days} causal historical days for its earliest query: '
            f'min_window_start={min_start}, required_backstep={historical_days * day_steps}'
        )

    print('SPRINT CANONICAL HISTORY SOURCE', {
        'speed_path': str(speed_path), 'speed_channel': speed_ch, 'observed_match_mae': speed_mae,
        'valid_path': str(valid_path), 'valid_channel': valid_ch, 'raw_valid_agreement': valid_agree,
        'historical_days': historical_days, 'context_days': source.context_days,
        'day_steps': day_steps, 'history_intervals': H,
        'causality': 'all historical timestamps < current query origin; current hidden/P12 future unavailable',
    }, flush=True)
    return source


def context_to_sprint_csv(path: Path, raw_context_with_nan: np.ndarray) -> None:
    """Write a real day x link x within-day-interval SPRINT input."""
    x = np.asarray(raw_context_with_nan, np.float32)
    if x.ndim != 3:
        raise ValueError(f'SPRINT context must be [D,H,N], got {x.shape}')
    D, H, N = map(int, x.shape)
    if D < 3 or H < 2:
        raise ValueError(f'SPRINT context is too small for historical-day adaptation: {x.shape}')
    index = pd.MultiIndex.from_tuples(
        [(d, t) for d in range(D) for t in range(H)], names=['date', 'time']
    )
    columns = [f'node_{j:05d}' for j in range(N)]
    # Preserve the compatibility behavior required by the official SPRINT parser:
    # observed values use one-decimal precision, while the two-decimal sentinel is missing.
    encoded = np.where(np.isfinite(x), np.round(x, 1), -999.99).astype(np.float32)
    pd.DataFrame(encoded.reshape(D * H, N), index=index, columns=columns).to_csv(
        path, float_format='%.2f'
    )


def sprint_result_to_current_query(x, days: int, H: int, N: int) -> np.ndarray:
    """Decode official SPRINT output and return only current/final-day [H,N]."""
    if isinstance(x, pd.DataFrame):
        if isinstance(x.columns, pd.MultiIndex):
            try:
                stacked = x.stack(level=-1, future_stack=True)
            except TypeError:
                stacked = x.stack(level=-1)
            try:
                stacked = stacked.sort_index()
            except Exception:
                pass
            a = stacked.to_numpy(dtype=np.float32)
            if a.size != days * H * N:
                raise RuntimeError(
                    f'SPRINT returned stacked DataFrame shape={a.shape}, expected {days*H*N} values'
                )
            a = a.reshape(days, H, N)
        else:
            a0 = x.to_numpy(dtype=np.float32)
            if a0.size != days * H * N:
                raise RuntimeError(
                    f'SPRINT returned DataFrame shape {a0.shape}, expected {days*H*N} values'
                )
            a = a0.reshape(days, N, H).transpose(0, 2, 1)
    else:
        a0 = np.asarray(x, dtype=np.float32)
        if a0.shape == (days, H, N):
            a = a0
        elif a0.shape == (days, N, H):
            a = a0.transpose(0, 2, 1)
        elif a0.size == days * H * N:
            a = a0.reshape(days, H, N)
        else:
            raise RuntimeError(
                f'SPRINT returned array shape {a0.shape}, expected {(days,H,N)} compatible output'
            )
    current = np.asarray(a[-1], np.float32)
    if current.shape != (H, N):
        raise RuntimeError(f'SPRINT decoded current query shape {current.shape}, expected {(H,N)}')
    if not np.isfinite(current).all():
        bad = int(np.size(current) - np.isfinite(current).sum())
        raise RuntimeError(f'SPRINT returned {bad} non-finite current-query values after decoding')
    return current
