from __future__ import annotations
import numpy as np


def restoration_scope_mask(shape, cfg):
    """Return the cells that the current SRD experiment is allowed to modify.

    Default: only the final `recent_steps` history rows. This keeps candidate
    quality and state-use quality aligned with the recent-suffix oracle.
    """
    H, N = int(shape[0]), int(shape[1])
    tcfg = (cfg or {}).get("transport", {})
    mode = str(tcfg.get("restoration_scope", "recent_suffix"))
    m = np.zeros((H, N), dtype=bool)
    if mode == "all_history":
        m[:] = True
        return m
    if mode != "recent_suffix":
        raise KeyError(f"Unknown restoration_scope={mode!r}; use recent_suffix or all_history")
    recent = max(1, min(H, int(tcfg.get("recent_steps", 3))))
    m[H-recent:H] = True
    return m


def restoration_scope_name(cfg, history: int | None = None):
    tcfg = (cfg or {}).get("transport", {})
    mode = str(tcfg.get("restoration_scope", "recent_suffix"))
    if mode == "all_history":
        return "all_history"
    recent = int(tcfg.get("recent_steps", 3))
    if history is not None:
        recent = max(1, min(int(history), recent))
    return f"recent{recent}"


def _masks(query, candidate, cfg):
    qobs = np.asarray(query["history_observed"], bool)
    qrest = np.asarray(query["restorable"], bool)
    cvalid = np.asarray(candidate["full_valid"], bool)
    scope = restoration_scope_mask(qrest.shape, cfg)
    overlap = qobs & cvalid
    fill = qrest & cvalid & scope
    return qobs, qrest, cvalid, overlap, fill


def raw_copy(query, candidate, cfg=None):
    x = np.asarray(query["x_obs"], np.float32).copy()
    _, _, _, _, fill = _masks(query, candidate, cfg)
    c = np.asarray(candidate["x_full"], np.float32)
    x[fill] = c[fill]
    return x, fill


def mean_residual(query, candidate, cfg=None):
    x = np.asarray(query["x_obs"], np.float32).copy()
    c = np.asarray(candidate["x_full"], np.float32)
    _, _, _, overlap, fill = _masks(query, candidate, cfg)
    _, N = x.shape
    for n in range(N):
        m = overlap[:, n]
        if m.sum() < 1:
            continue
        shift = float(x[m, n].mean() - c[m, n].mean())
        f = fill[:, n]
        x[f, n] = c[f, n] + shift
    return x, fill


def zscore(query, candidate, cfg=None):
    x = np.asarray(query["x_obs"], np.float32).copy()
    c = np.asarray(candidate["x_full"], np.float32)
    _, _, _, overlap, fill = _masks(query, candidate, cfg)
    eps = float((cfg or {}).get("transport", {}).get("eps", 1e-6))
    clip = (cfg or {}).get("transport", {}).get("scale_clip", [0.5, 2.0])
    _, N = x.shape
    for n in range(N):
        m = overlap[:, n]
        if m.sum() < 2:
            continue
        qm, cm = float(x[m, n].mean()), float(c[m, n].mean())
        qs, cs = float(x[m, n].std()), float(c[m, n].std())
        scale = np.clip(qs / (cs + eps), float(clip[0]), float(clip[1]))
        f = fill[:, n]
        x[f, n] = qm + scale * (c[f, n] - cm)
    return x, fill


def delta_anchor(query, candidate, cfg=None):
    x = np.asarray(query["x_obs"], np.float32).copy()
    c = np.asarray(candidate["x_full"], np.float32)
    qobs, _, cvalid, _, fill = _masks(query, candidate, cfg)
    _, N = x.shape
    for n in range(N):
        anchors = np.flatnonzero(qobs[:, n] & cvalid[:, n])
        if anchors.size == 0:
            continue
        for t in np.flatnonzero(fill[:, n]):
            # Prefer the latest causal observed anchor. If one does not exist,
            # fall back to the closest observed anchor so the diagnostic can
            # still quantify support availability.
            prev = anchors[anchors < t]
            a = int(prev[-1]) if prev.size else int(anchors[np.argmin(np.abs(anchors - t))])
            x[t, n] = x[a, n] + (c[t, n] - c[a, n])
    return x, fill


METHODS = {
    "raw_copy": raw_copy,
    "mean_residual": mean_residual,
    "zscore": zscore,
    "delta_anchor": delta_anchor,
}


def apply_transport(name, query, candidate, cfg):
    if name not in METHODS:
        raise KeyError(f"Unknown transport {name}; choose {sorted(METHODS)}")
    return METHODS[name](query, candidate, cfg)
