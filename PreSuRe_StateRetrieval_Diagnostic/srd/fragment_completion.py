from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .patch_transport import patch_overlap_mask, _affine_params, suffix_mask


@dataclass(frozen=True)
class FragmentSpec:
    fragment_id: str
    mask: np.ndarray
    time_start: int
    time_end: int
    node_count: int
    cell_count: int


def _patch_mask(shape, nodes):
    m = np.zeros(shape, dtype=bool)
    m[:, np.asarray(nodes, np.int64)] = True
    return m


def _time_blocks(H: int, recent_steps: int, block_size: int):
    L = max(1, min(int(H), int(recent_steps)))
    start = H - L
    b = max(1, int(block_size))
    out = []
    s = start
    while s < H:
        e = min(H, s + b)
        out.append((s, e))
        s = e
    return out


def build_fragments(query, nodes, recent_steps: int, geometry: str, block_size: int = 3):
    """Partition the *restorable* cells of one PatchSTG patch into fragments.

    All returned masks are disjoint and are subsets of the requested recent suffix.
    Geometry options:
      - whole_patch: one fragment for the whole damaged patch/suffix
      - temporal: one patch-wide fragment per temporal block
      - node_track: one node-wide fragment across the suffix
      - node_block: one node x temporal-block fragment
      - cell: one fragment per restorable cell (diagnostic ceiling)
    """
    rest = np.asarray(query["restorable"], bool)
    H, _ = rest.shape
    nodes = np.asarray(nodes, np.int64)
    scope = suffix_mask(rest.shape, recent_steps)
    target = rest & scope & _patch_mask(rest.shape, nodes)
    if not np.any(target):
        return []

    geometry = str(geometry)
    specs: list[FragmentSpec] = []

    def add(fid, m):
        m = np.asarray(m, bool) & target
        if not np.any(m):
            return
        ts = np.flatnonzero(m.any(axis=1))
        ns = np.flatnonzero(m.any(axis=0))
        specs.append(FragmentSpec(
            fragment_id=str(fid), mask=m,
            time_start=int(ts.min()), time_end=int(ts.max()) + 1,
            node_count=int(len(ns)), cell_count=int(m.sum()),
        ))

    if geometry == "whole_patch":
        add("whole", target)
    elif geometry == "temporal":
        for bi, (s, e) in enumerate(_time_blocks(H, recent_steps, block_size)):
            m = np.zeros_like(target); m[s:e, :] = True
            add(f"t{bi}:{s}-{e}", m)
    elif geometry == "node_track":
        for n in nodes:
            m = np.zeros_like(target); m[:, int(n)] = True
            add(f"n{int(n)}", m)
    elif geometry == "node_block":
        for n in nodes:
            for bi, (s, e) in enumerate(_time_blocks(H, recent_steps, block_size)):
                m = np.zeros_like(target); m[s:e, int(n)] = True
                add(f"n{int(n)}_t{bi}:{s}-{e}", m)
    elif geometry == "cell":
        for t, n in np.argwhere(target):
            m = np.zeros_like(target); m[int(t), int(n)] = True
            add(f"c{int(t)}_{int(n)}", m)
    else:
        raise KeyError(f"Unknown fragment geometry={geometry}")

    # Strict partition check: fragments must cover the target exactly once.
    cover = np.zeros_like(target, dtype=np.int16)
    for s in specs:
        cover += s.mask.astype(np.int16)
    if not np.array_equal(cover > 0, target) or int(cover.max(initial=0)) > 1:
        raise RuntimeError(f"Fragment partition failure geometry={geometry}")
    return specs


def adapted_candidate_patch(query, candidate, nodes, method: str, cfg):
    """Return adapted candidate values and validity for one patch, without using hidden query truth."""
    nodes = np.asarray(nodes, np.int64)
    qx = np.asarray(query["x_obs"], np.float32)
    cx = np.asarray(candidate["x_full"], np.float32)
    cv = np.asarray(candidate["full_valid"], bool)
    vals = cx.copy()
    method = str(method)

    if method == "raw_copy":
        pass
    elif method == "mean_shift":
        overlap = patch_overlap_mask(query, candidate, nodes, str(cfg["patch_transport"].get("alignment_scope", "all")))
        shift = float(qx[overlap].mean() - cx[overlap].mean()) if np.any(overlap) else 0.0
        vals = cx + shift
    elif method == "affine_alignment":
        overlap = patch_overlap_mask(query, candidate, nodes, str(cfg["patch_transport"].get("alignment_scope", "all")))
        a, b = _affine_params(query, candidate, overlap, cfg)
        vals = a * cx + b
    elif method == "delta_anchor":
        # For fragment diagnostics we need a dense candidate state. Delta-anchor is inherently
        # target-cell dependent, so use the candidate level here for scoring and leave its exact
        # transport to the legacy transport diagnostic.
        vals = cx.copy()
    else:
        raise KeyError(method)

    pm = _patch_mask(qx.shape, nodes)
    return vals.astype(np.float32), (cv & pm)


def build_candidate_bank(query, train, candidate_ids, nodes, method: str, cfg):
    vals, valid, ids = [], [], []
    for cid in np.asarray(candidate_ids, np.int64):
        if int(cid) < 0:
            continue
        c = train.row(int(cid))
        v, m = adapted_candidate_patch(query, c, nodes, method, cfg)
        vals.append(v); valid.append(m); ids.append(int(cid))
    if not vals:
        H, N = np.asarray(query["x_obs"]).shape
        return (np.empty((0, H, N), np.float32), np.empty((0, H, N), bool), np.empty((0,), np.int64))
    return np.stack(vals), np.stack(valid), np.asarray(ids, np.int64)


def fragment_trial_errors(query, cand_values, cand_valid, fragment_mask):
    """Completion MAE per candidate, counting uncovered target cells as unchanged baseline cells."""
    truth = np.asarray(query["x_full"], np.float32)
    base = np.asarray(query["x_obs"], np.float32)
    target = np.asarray(fragment_mask, bool) & np.isfinite(truth)
    if not np.any(target) or len(cand_values) == 0:
        return np.full((len(cand_values),), np.inf, np.float64), np.zeros((len(cand_values),), np.float64)
    tv = truth[target]
    bv = base[target]
    cv = np.asarray(cand_values[:, target], np.float32)
    vm = np.asarray(cand_valid[:, target], bool) & np.isfinite(cv)
    trial = np.where(vm, cv, bv[None, :])
    err = np.mean(np.abs(trial - tv[None, :]), axis=1, dtype=np.float64)
    coverage = vm.mean(axis=1, dtype=np.float64)
    return err, coverage


def fragment_anchor_mask(query, nodes, fragment: FragmentSpec, context_radius: int = 2, min_anchor_cells: int = 4):
    """Observed local context used to score a fragment without hidden values."""
    obs = np.asarray(query["history_observed"], bool)
    H, _ = obs.shape
    nodes = np.asarray(nodes, np.int64)
    pm = _patch_mask(obs.shape, nodes)
    s = max(0, int(fragment.time_start) - int(context_radius))
    e = min(H, int(fragment.time_end) + int(context_radius))
    tm = np.zeros_like(obs); tm[s:e, :] = True
    anchor = obs & pm & tm
    if int(anchor.sum()) < int(min_anchor_cells):
        anchor = obs & pm
    return anchor


def observable_fragment_scores(query, cand_values, cand_valid, anchor_mask, level_weight=0.75, delta_weight=0.25, min_anchor_cells=4):
    """Candidate score using only observed local context around the target fragment."""
    qx = np.asarray(query["x_obs"], np.float32)
    anchor = np.asarray(anchor_mask, bool)
    K = len(cand_values)
    if K == 0 or int(anchor.sum()) < int(min_anchor_cells):
        return np.full((K,), np.inf, np.float64)

    common = np.asarray(cand_valid, bool) & anchor[None, :, :]
    n = common.sum(axis=(1, 2))
    d = np.abs(np.asarray(cand_values, np.float32) - qx[None, :, :])
    num = np.where(common, d, 0.0).sum(axis=(1, 2), dtype=np.float64)
    level = np.divide(num, n, out=np.full(K, np.inf, np.float64), where=n >= int(min_anchor_cells))

    if qx.shape[0] >= 2:
        qd = qx[1:] - qx[:-1]
        qdm = anchor[1:] & anchor[:-1]
        cd = cand_values[:, 1:] - cand_values[:, :-1]
        cm = cand_valid[:, 1:] & cand_valid[:, :-1] & qdm[None, :, :]
        dn = cm.sum(axis=(1, 2))
        dd = np.abs(cd - qd[None, :, :])
        dnum = np.where(cm, dd, 0.0).sum(axis=(1, 2), dtype=np.float64)
        delta = np.divide(dnum, dn, out=level.copy(), where=dn >= max(1, int(min_anchor_cells) // 2))
    else:
        delta = level.copy()
    return float(level_weight) * level + float(delta_weight) * delta


def choose_fragment_candidate(query, cand_values, cand_valid, fragment: FragmentSpec, nodes, cfg, mode: str,
                              context_radius=2, min_anchor_cells=4, min_coverage=0.8):
    """Return (index, diagnostics) for one fragment.

    `oracle_completion` uses hidden history truth and is diagnostic-only.
    `local_observed` uses only observed query context and candidate history.
    `global_top1` uses the original patch retrieval order.
    """
    errs, coverage = fragment_trial_errors(query, cand_values, cand_valid, fragment.mask)
    eligible = coverage >= float(min_coverage)
    mode = str(mode)
    local_scores = np.full_like(errs, np.inf, dtype=np.float64)
    if mode == "oracle_completion":
        score = np.where(eligible, errs, np.inf)
    elif mode == "local_observed":
        anchor = fragment_anchor_mask(query, nodes, fragment, context_radius, min_anchor_cells)
        local_scores = observable_fragment_scores(
            query, cand_values, cand_valid, anchor,
            level_weight=float(cfg["fragment_diagnostic"].get("local_level_weight", 0.75)),
            delta_weight=float(cfg["fragment_diagnostic"].get("local_delta_weight", 0.25)),
            min_anchor_cells=min_anchor_cells,
        )
        score = np.where(eligible, local_scores, np.inf)
    elif mode == "global_top1":
        score = np.arange(len(errs), dtype=np.float64)
        score = np.where(eligible, score, np.inf)
    else:
        raise KeyError(mode)
    if not np.any(np.isfinite(score)):
        return None, {"oracle_error": float("inf"), "coverage": 0.0, "local_score": float("inf")}
    idx = int(np.nanargmin(score))
    return idx, {
        "oracle_error": float(errs[idx]), "coverage": float(coverage[idx]),
        "local_score": float(local_scores[idx]) if np.isfinite(local_scores[idx]) else float("inf"),
    }


def compose_fragment_selection(query, train, candidate_ids, nodes, recent_steps: int, geometry: str, block_size: int,
                               method: str, cfg, selection_mode: str):
    """Compose one patch from candidate fragments and return completed input + diagnostics."""
    qx = np.asarray(query["x_obs"], np.float32)
    fragments = build_fragments(query, nodes, recent_steps, geometry, block_size)
    if not fragments:
        return qx.copy(), {
            "fragments": 0, "selected_fragments": 0, "unique_sources": 0,
            "mean_selected_rank": float("nan"), "mean_coverage": float("nan"),
            "oracle_rank1_hit_rate": float("nan"), "oracle_top5_hit_rate": float("nan"),
        }, []
    vals, valid, ids = build_candidate_bank(query, train, candidate_ids, nodes, method, cfg)
    K = len(ids)
    out = qx.copy()
    details = []
    ranks, coverages, selected_ids = [], [], []
    oracle_hit1, oracle_hit5 = [], []
    fcfg = cfg["fragment_diagnostic"]
    for frag in fragments:
        errs, cov = fragment_trial_errors(query, vals, valid, frag.mask)
        oracle_order = np.argsort(errs, kind="stable") if len(errs) else np.empty((0,), np.int64)
        oracle_best = int(oracle_order[0]) if len(oracle_order) else -1
        idx, meta = choose_fragment_candidate(
            query, vals, valid, frag, nodes, cfg, selection_mode,
            context_radius=int(fcfg.get("context_radius", 2)),
            min_anchor_cells=int(fcfg.get("min_anchor_cells", 4)),
            min_coverage=float(fcfg.get("min_fragment_coverage", 0.8)),
        )
        if idx is None:
            details.append({"fragment_id": frag.fragment_id, "selected": False})
            continue
        if selection_mode == "oracle_completion" and bool(fcfg.get("allow_oracle_abstain", True)):
            truth = np.asarray(query["x_full"], np.float32)
            base = np.asarray(query["x_obs"], np.float32)
            tm = frag.mask & np.isfinite(truth)
            base_err = float(np.abs(base[tm] - truth[tm]).mean()) if np.any(tm) else float("inf")
            if not np.isfinite(meta["oracle_error"]) or float(meta["oracle_error"]) >= base_err:
                details.append({"fragment_id": frag.fragment_id, "selected": False, "oracle_abstain": True, "baseline_completion_mae": base_err})
                continue
        fm = frag.mask & valid[idx]
        out[fm] = vals[idx][fm]
        rank = int(idx + 1)  # original global patch retrieval rank among the truncated pool
        ranks.append(rank); coverages.append(float(meta["coverage"])); selected_ids.append(int(ids[idx]))
        oracle_hit1.append(float(idx == oracle_best))
        oracle_hit5.append(float(idx in set(oracle_order[:min(5, len(oracle_order))].tolist())))
        details.append({
            "fragment_id": frag.fragment_id, "selected": True, "candidate_id": int(ids[idx]),
            "candidate_rank": rank, "fragment_cells": int(frag.cell_count),
            "coverage": float(meta["coverage"]), "selected_completion_mae": float(meta["oracle_error"]),
            "oracle_best_rank": int(oracle_best + 1) if oracle_best >= 0 else 0,
        })
    return out, {
        "fragments": int(len(fragments)), "selected_fragments": int(len(ranks)),
        "unique_sources": int(len(set(selected_ids))),
        "mean_selected_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "mean_coverage": float(np.mean(coverages)) if coverages else float("nan"),
        "oracle_rank1_hit_rate": float(np.mean(oracle_hit1)) if oracle_hit1 else float("nan"),
        "oracle_top5_hit_rate": float(np.mean(oracle_hit5)) if oracle_hit5 else float("nan"),
    }, details


def compose_fragment_softmax(query, train, candidate_ids, nodes, recent_steps: int, geometry: str, block_size: int,
                             method: str, cfg, temperature: float = 0.35):
    """Realizable fragment-wise softmax composition using only local observed-context scores."""
    qx = np.asarray(query["x_obs"], np.float32)
    fragments = build_fragments(query, nodes, recent_steps, geometry, block_size)
    vals, valid, ids = build_candidate_bank(query, train, candidate_ids, nodes, method, cfg)
    out = qx.copy(); fcfg = cfg["fragment_diagnostic"]
    source_mass = np.zeros(len(ids), np.float64)
    used = 0
    for frag in fragments:
        anchor = fragment_anchor_mask(query, nodes, frag, int(fcfg.get("context_radius", 2)), int(fcfg.get("min_anchor_cells", 4)))
        sc = observable_fragment_scores(
            query, vals, valid, anchor,
            level_weight=float(fcfg.get("local_level_weight", 0.75)),
            delta_weight=float(fcfg.get("local_delta_weight", 0.25)),
            min_anchor_cells=int(fcfg.get("min_anchor_cells", 4)),
        )
        _, cov = fragment_trial_errors(query, vals, valid, frag.mask)
        ok = np.isfinite(sc) & (cov >= float(fcfg.get("min_fragment_coverage", 0.8)))
        if not np.any(ok):
            continue
        d0 = float(np.min(sc[ok])); temp = max(float(temperature), 1e-6)
        w = np.zeros(len(ids), np.float64); w[ok] = np.exp(-(sc[ok] - d0) / temp)
        target_idx = np.argwhere(frag.mask)
        for t, n in target_idx:
            vm = ok & valid[:, int(t), int(n)] & np.isfinite(vals[:, int(t), int(n)])
            if not np.any(vm):
                continue
            ww = w[vm]
            if float(ww.sum()) <= 0:
                continue
            ww = ww / ww.sum()
            out[int(t), int(n)] = float(np.sum(ww * vals[vm, int(t), int(n)]))
            source_mass[np.flatnonzero(vm)] += ww
        used += 1
    eff_sources = int((source_mass > 1e-6).sum()) if len(source_mass) else 0
    return out, {"fragments": int(len(fragments)), "used_fragments": int(used), "effective_sources": eff_sources}
