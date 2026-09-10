from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph import local_patch
from .utils import read_json, resolve_rel


@dataclass(frozen=True)
class MemoryEpisode:
    episode_id: int
    window_start: int
    history_start: int
    history_end: int
    future_end: int


def build_static_patches(adj: np.ndarray, hops: int, max_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    n = adj.shape[0]
    ids = np.full((n, max_nodes), -1, dtype=np.int64)
    valid = np.zeros((n, max_nodes), dtype=np.uint8)
    for target in range(n):
        p = local_patch(adj, target, hops=hops, max_nodes=max_nodes)
        ids[target, :len(p)] = p
        valid[target, :len(p)] = 1
    return ids, valid


def candidate_allowed(query_start: int, candidate_start: int, H: int, P: int, strict: bool = True) -> bool:
    if not strict:
        return int(candidate_start) != int(query_start)
    candidate_future_end = int(candidate_start) + H + P
    return candidate_future_end <= int(query_start)
