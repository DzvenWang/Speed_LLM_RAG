from __future__ import annotations

import numpy as np


def chronological_splits(T: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict:
    total = float(train_ratio + val_ratio + test_ratio)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f'Split ratios must sum to 1.0, got {total}')
    b1 = int(np.floor(train_ratio * T))
    b2 = int(np.floor((train_ratio + val_ratio) * T))
    return {
        'train': {'start': 0, 'end_exclusive': b1, 'n_steps': b1},
        'val': {'start': b1, 'end_exclusive': b2, 'n_steps': b2-b1},
        'test': {'start': b2, 'end_exclusive': T, 'n_steps': T-b2},
    }


def valid_window_starts(start: int, end: int, history: int, horizon: int, stride: int) -> np.ndarray:
    last = int(end) - int(history) - int(horizon)
    if last < int(start):
        return np.empty(0, dtype=np.int64)
    return np.arange(int(start), last + 1, int(stride), dtype=np.int64)


def filter_complete_windows(
    starts: np.ndarray,
    valid_speed: np.ndarray,
    history: int,
    horizon: int,
    min_fraction: float = 1.0,
) -> np.ndarray:
    """Filter windows using a rolling count over time, avoiding per-window tensor scans."""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return starts
    v = np.asarray(valid_speed, dtype=bool)
    span = int(history) + int(horizon)
    # Number of valid node observations at each timestamp.
    step_valid = v.sum(axis=1, dtype=np.int64)
    prefix = np.concatenate([[0], np.cumsum(step_valid, dtype=np.int64)])
    total = span * v.shape[1]
    counts = prefix[starts + span] - prefix[starts]
    keep = counts.astype(np.float64) / max(total, 1) + 1e-12 >= float(min_fraction)
    return starts[keep]
