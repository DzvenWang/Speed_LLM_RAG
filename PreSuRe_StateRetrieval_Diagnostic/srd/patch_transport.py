from __future__ import annotations

import numpy as np


def suffix_mask(shape, recent_steps):
    H, N = int(shape[0]), int(shape[1])
    L = max(1, min(H, int(recent_steps)))
    m = np.zeros((H, N), dtype=bool)
    m[H - L:H] = True
    return m


def _candidate_arrays(candidate):
    c = np.asarray(candidate["x_full"], np.float32)
    cv = np.asarray(candidate["full_valid"], bool)
    return c, cv


def patch_fill_mask(query, candidate, nodes, recent_steps):
    nodes = np.asarray(nodes, np.int64)
    qrest = np.asarray(query["restorable"], bool)
    _, cv = _candidate_arrays(candidate)
    scope = suffix_mask(qrest.shape, recent_steps)
    node_mask = np.zeros_like(qrest)
    node_mask[:, nodes] = True
    return qrest & cv & scope & node_mask


def patch_overlap_mask(query, candidate, nodes, alignment_scope="all"):
    nodes = np.asarray(nodes, np.int64)
    qobs = np.asarray(query["history_observed"], bool)
    _, cv = _candidate_arrays(candidate)
    m = qobs & cv
    node_mask = np.zeros_like(m)
    node_mask[:, nodes] = True
    m &= node_mask
    if alignment_scope != "all":
        if not str(alignment_scope).startswith("recent"):
            raise KeyError(f"Unknown alignment_scope={alignment_scope}")
        L = int(str(alignment_scope).replace("recent", ""))
        m &= suffix_mask(m.shape, L)
    return m


def _affine_params(query, candidate, overlap, cfg):
    qx = np.asarray(query["x_obs"], np.float32)
    cx = np.asarray(candidate["x_full"], np.float32)
    qv = qx[overlap].astype(np.float64)
    cv = cx[overlap].astype(np.float64)
    min_pts = int(cfg["patch_transport"].get("affine_min_points", 4))
    if qv.size < min_pts:
        if qv.size:
            return 1.0, float(qv.mean() - cv.mean())
        return 1.0, 0.0
    cm, qm = float(cv.mean()), float(qv.mean())
    var = float(np.mean((cv - cm) ** 2))
    eps = float(cfg["patch_transport"].get("eps", 1e-6))
    if var <= eps:
        a = 1.0
    else:
        a = float(np.mean((cv - cm) * (qv - qm)) / (var + eps))
    clip = cfg["patch_transport"].get("affine_scale_clip", [0.5, 2.0])
    a = float(np.clip(a, float(clip[0]), float(clip[1])))
    b = float(qm - a * cm)
    return a, b


def candidate_proposal(query, candidate, nodes, recent_steps, method, cfg):
    """Return (proposal, fill_mask) for exactly one PatchSTG patch.

    The proposal has full [H,N] shape for simple composition, but only `fill_mask`
    entries are legal interventions. Observed cells are never modified.
    """
    qx = np.asarray(query["x_obs"], np.float32)
    cx, _ = _candidate_arrays(candidate)
    fill = patch_fill_mask(query, candidate, nodes, recent_steps)
    out = qx.copy()
    if not np.any(fill):
        return out, fill

    method = str(method)
    if method == "raw_copy":
        out[fill] = cx[fill]
        return out, fill

    align_scope = str(cfg["patch_transport"].get("alignment_scope", "all"))
    overlap = patch_overlap_mask(query, candidate, nodes, align_scope)

    if method == "mean_shift":
        if np.any(overlap):
            shift = float(qx[overlap].mean() - cx[overlap].mean())
        else:
            shift = 0.0
        out[fill] = cx[fill] + shift
        return out, fill

    if method == "affine_alignment":
        a, b = _affine_params(query, candidate, overlap, cfg)
        out[fill] = a * cx[fill] + b
        return out, fill

    if method == "delta_anchor":
        nodes = np.asarray(nodes, np.int64)
        qobs = np.asarray(query["history_observed"], bool)
        cvalid = np.asarray(candidate["full_valid"], bool)
        for n in nodes:
            anchors = np.flatnonzero(qobs[:, n] & cvalid[:, n])
            if anchors.size == 0:
                continue
            for t in np.flatnonzero(fill[:, n]):
                prev = anchors[anchors < t]
                a = int(prev[-1]) if prev.size else int(anchors[np.argmin(np.abs(anchors - t))])
                out[t, n] = qx[a, n] + (cx[t, n] - cx[a, n])
        return out, fill

    raise KeyError(f"Unknown patch transport method={method}")


def inject_patch(base_x, proposal, fill_mask):
    out = np.asarray(base_x, np.float32).copy()
    f = np.asarray(fill_mask, bool)
    out[f] = np.asarray(proposal, np.float32)[f]
    return out


def completion_mae(query, proposal, fill_mask):
    truth = np.asarray(query["x_full"], np.float32)
    f = np.asarray(fill_mask, bool) & np.isfinite(truth) & np.isfinite(proposal)
    if not np.any(f):
        return float("inf")
    return float(np.abs(np.asarray(proposal, np.float32)[f] - truth[f]).mean())


def weighted_topk_proposal(query, candidates, distances, nodes, recent_steps, cfg, base_method="affine_alignment"):
    """Similarity-weighted support composition for one patch.

    Each candidate is first adapted to the current state using `base_method`; values are
    then averaged only over candidates valid at each missing cell.
    """
    qx = np.asarray(query["x_obs"], np.float32)
    accum = np.zeros_like(qx, dtype=np.float64)
    weight = np.zeros_like(qx, dtype=np.float64)
    distances = np.asarray(distances, np.float64)
    temp = max(float(cfg["patch_transport"].get("weighted_temperature", 0.35)), 1e-6)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return qx.copy(), np.zeros_like(qx, bool)
    d0 = float(np.min(distances[finite]))
    ws = np.zeros_like(distances, dtype=np.float64)
    ws[finite] = np.exp(-(distances[finite] - d0) / temp)
    for cand, w in zip(candidates, ws):
        if w <= 0:
            continue
        p, f = candidate_proposal(query, cand, nodes, recent_steps, base_method, cfg)
        accum[f] += float(w) * p[f]
        weight[f] += float(w)
    out = qx.copy()
    fill = weight > 0
    out[fill] = (accum[fill] / weight[fill]).astype(np.float32)
    return out, fill
