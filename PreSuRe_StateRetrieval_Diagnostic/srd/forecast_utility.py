from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class PatchAction:
    patch_id: int
    rank: int
    candidate_id: int
    forecast_mae: float
    forecast_utility: float
    completion_mae: float = float("nan")


def select_best_action_by_patch(actions: Iterable[PatchAction], k: int, baseline_mae: float, allow_abstain: bool = True):
    """Select the lowest-forecast-MAE action per patch within the first k retrieval ranks.

    This helper is diagnostic-only when `forecast_mae` was measured against the true future.
    If `allow_abstain` is true, actions that do not beat the baseline are omitted.
    """
    k = max(1, int(k))
    best = {}
    for a in actions:
        if int(a.rank) > k or not np.isfinite(a.forecast_mae):
            continue
        cur = best.get(int(a.patch_id))
        if cur is None or float(a.forecast_mae) < float(cur.forecast_mae):
            best[int(a.patch_id)] = a
    if allow_abstain:
        best = {pid: a for pid, a in best.items() if float(a.forecast_mae) < float(baseline_mae)}
    return best


def best_single_action(actions: Iterable[PatchAction], k: int, baseline_mae: float, allow_abstain: bool = True):
    eligible = [a for a in actions if int(a.rank) <= int(k) and np.isfinite(a.forecast_mae)]
    if not eligible:
        return None
    a = min(eligible, key=lambda x: float(x.forecast_mae))
    if allow_abstain and float(a.forecast_mae) >= float(baseline_mae):
        return None
    return a


def build_same_episode_pool(candidate_ids, scores, eligible, qpos: int, max_candidates: int | None = None):
    """Build a coherent episode pool from patch-local retrieval results.

    The ranking is inference-available: it uses only the patch-local retrieval ranks/scores.
    A train episode can enter the pool if it appears in at least one eligible patch's Top-K.
    Returned rows are sorted by a coverage-penalized rank score (lower is better).
    """
    ids = np.asarray(candidate_ids)
    sc = np.asarray(scores)
    el = np.asarray(eligible, bool)
    if ids.ndim != 3 or sc.shape != ids.shape:
        raise ValueError("candidate_ids/scores must have shape [Q,P,K]")
    qpos = int(qpos)
    K = int(ids.shape[-1])
    pids = np.flatnonzero(el[qpos])
    P = int(len(pids))
    if P == 0:
        return []

    by_id = {}
    for pid in pids:
        for r in range(K):
            cid = int(ids[qpos, pid, r])
            if cid < 0:
                continue
            row = by_id.setdefault(cid, {"candidate_id": cid, "ranks": [], "scores": [], "patches": []})
            row["ranks"].append(int(r + 1))
            sval = float(sc[qpos, pid, r])
            if np.isfinite(sval):
                row["scores"].append(sval)
            row["patches"].append(int(pid))

    rows = []
    for cid, row in by_id.items():
        ranks = np.asarray(row["ranks"], np.float64)
        hits = int(len(set(row["patches"])))
        coverage = float(hits / P)
        mean_rank_fraction = float(np.mean(ranks / max(K, 1))) if ranks.size else 1.0
        # Missing patch coverage is deliberately penalized. This ranking is only a
        # realizable ordering/cap mechanism; oracle selection happens after forecasting.
        rank_score = float((1.0 - coverage) + mean_rank_fraction)
        rows.append({
            "candidate_id": int(cid),
            "min_rank": int(np.min(ranks)) if ranks.size else K + 1,
            "mean_rank": float(np.mean(ranks)) if ranks.size else float("inf"),
            "patch_hits": hits,
            "coverage": coverage,
            "episode_rank_score": rank_score,
            "mean_local_score": float(np.mean(row["scores"])) if row["scores"] else float("inf"),
        })
    rows.sort(key=lambda r: (r["min_rank"], r["episode_rank_score"], r["mean_local_score"], r["candidate_id"]))
    if max_candidates is not None and int(max_candidates) > 0:
        rows = rows[: int(max_candidates)]
    return rows
