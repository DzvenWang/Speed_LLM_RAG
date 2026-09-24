from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PatchRetrievalResult:
    candidate_ids: np.ndarray
    score: np.ndarray
    level_mae: np.ndarray
    delta_mae: np.ndarray
    overlap_fraction: np.ndarray
    tod_distance: np.ndarray
    dow_match: np.ndarray


def _candidate_valid(train, ids, nodes):
    obs = np.asarray(train.history_observed[ids][:, :, nodes], bool)
    artificial = np.asarray(train.artificial_missing[ids][:, :, nodes], bool)
    xf = np.asarray(train.x_full_norm[ids][:, :, nodes], np.float32)
    return obs | (artificial & np.isfinite(xf)), xf


def _masked_mae_batch(cand, query, mask):
    # cand: [B,T,P], query/mask: [T,P]
    m = np.asarray(mask, bool)[None, :, :]
    n = np.broadcast_to(m, cand.shape).sum(axis=(1, 2))
    d = np.abs(cand - query[None, :, :])
    num = np.where(m, d, 0.0).sum(axis=(1, 2), dtype=np.float64)
    return np.divide(num, n, out=np.full(len(cand), np.inf, dtype=np.float64), where=n > 0), n


def _local_distances(q, train, ids, nodes, cfg):
    nodes = np.asarray(nodes, np.int64)
    qx = np.asarray(q["x_obs"][:, nodes], np.float32)
    qobs = np.asarray(q["history_observed"][:, nodes], bool)
    cvalid, cx = _candidate_valid(train, ids, nodes)

    common = cvalid & qobs[None, :, :]
    n = common.sum(axis=(1, 2))
    qn = max(int(qobs.sum()), 1)
    overlap = n.astype(np.float64) / float(qn)

    d = np.abs(cx - qx[None, :, :])
    num = np.where(common, d, 0.0).sum(axis=(1, 2), dtype=np.float64)
    level = np.divide(num, n, out=np.full(len(ids), np.inf, dtype=np.float64), where=n > 0)

    # Local trajectory compatibility. Only differences whose two query endpoints are
    # observed are legal retrieval evidence.
    if qx.shape[0] >= 2:
        qdelta = qx[1:] - qx[:-1]
        qdmask = qobs[1:] & qobs[:-1]
        cdelta = cx[:, 1:] - cx[:, :-1]
        cdvalid = cvalid[:, 1:] & cvalid[:, :-1]
        dcommon = cdvalid & qdmask[None, :, :]
        dn = dcommon.sum(axis=(1, 2))
        dd = np.abs(cdelta - qdelta[None, :, :])
        dnum = np.where(dcommon, dd, 0.0).sum(axis=(1, 2), dtype=np.float64)
        delta = np.divide(dnum, dn, out=level.copy(), where=dn > 0)
    else:
        delta = level.copy()

    min_overlap = float(cfg["patch_retrieval"].get("min_observed_overlap", 0.50))
    level = np.where(overlap >= min_overlap, level, np.inf)
    delta = np.where(overlap >= min_overlap, delta, np.inf)
    return level, delta, overlap


def retrieve_patch(q, train, train_ids, nodes, cfg):
    """Mask-conditioned retrieval for one real PatchSTG spatial patch.

    Candidate ranking sees only cells observed in the query. Hidden/current ground-truth
    values are never used here. All candidates come from the train split.
    """
    rcfg = cfg["patch_retrieval"]
    train_ids = np.asarray(train_ids, np.int64)
    nodes = np.asarray(nodes, np.int64)
    qobs = np.asarray(q["history_observed"][:, nodes], bool)
    if int(qobs.sum()) < int(rcfg.get("min_query_observed_cells", 4)):
        return PatchRetrievalResult(
            *(np.empty((0,), dtype=d) for d in [np.int64, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])
        )

    chunk = int(rcfg.get("distance_chunk_size", 512))
    level_parts, delta_parts, overlap_parts = [], [], []
    for s in range(0, len(train_ids), chunk):
        ids = train_ids[s:s + chunk]
        lv, de, ov = _local_distances(q, train, ids, nodes, cfg)
        level_parts.append(lv)
        delta_parts.append(de)
        overlap_parts.append(ov)
    level = np.concatenate(level_parts)
    delta = np.concatenate(delta_parts)
    overlap = np.concatenate(overlap_parts)

    # Time context is a small optional prior, never a substitute for local state match.
    mtod = np.asarray(train.tod_hist[train_ids, -1], np.int64)
    mdow = np.asarray(train.dow_hist[train_ids, -1], np.int64)
    qtod = int(q["tod_hist"][-1]); qdow = int(q["dow_hist"][-1])
    period_cfg = int(rcfg.get("tod_period", 0))
    if period_cfg > 1:
        period = period_cfg
    else:
        period = max(int(np.max(mtod)) + 1 if mtod.size else 2, qtod + 1, 2)
    dt = np.abs(mtod.astype(np.float64) - float(qtod))
    dt = np.minimum(dt, np.maximum(float(period) - dt, 0.0))
    tod_distance = dt / max(float(period) / 2.0, 1.0)
    dow_match = (mdow == qdow).astype(np.float64)

    lw = float(rcfg.get("level_weight", 0.75))
    dw = float(rcfg.get("delta_weight", 0.25))
    tw = float(rcfg.get("tod_penalty_weight", 0.0))
    ww = float(rcfg.get("dow_penalty_weight", 0.0))
    score = lw * level + dw * delta + tw * tod_distance + ww * (1.0 - dow_match)

    finite = np.flatnonzero(np.isfinite(score))
    if finite.size == 0:
        return PatchRetrievalResult(
            *(np.empty((0,), dtype=d) for d in [np.int64, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])
        )
    top_k = min(int(rcfg.get("top_k", 50)), finite.size)
    local = finite[np.argpartition(score[finite], top_k - 1)[:top_k]]
    local = local[np.argsort(score[local], kind="stable")]
    return PatchRetrievalResult(
        candidate_ids=train_ids[local].astype(np.int64),
        score=score[local].astype(np.float32),
        level_mae=level[local].astype(np.float32),
        delta_mae=delta[local].astype(np.float32),
        overlap_fraction=overlap[local].astype(np.float32),
        tod_distance=tod_distance[local].astype(np.float32),
        dow_match=dow_match[local].astype(np.float32),
    )
