from __future__ import annotations
import numpy as np
from .evaluate import predict_raw, pooled_prehead


def _context_score(q_tod, q_dow, c_tod, c_dow, cfg):
    period = max(int(np.max(c_tod)) + 1, int(q_tod) + 1, 2)
    dt = np.abs(np.asarray(c_tod, dtype=np.float32) - float(q_tod))
    dt = np.minimum(dt, period - dt)
    tod = 0.5 * (1.0 + np.cos(2 * np.pi * dt / period))
    dow = (np.asarray(c_dow) == int(q_dow)).astype(np.float32)
    return float(cfg["retrieval"]["tod_weight"]) * tod + float(cfg["retrieval"]["dow_weight"]) * dow


def _scope_stats(query_x, query_obs, cand_x, cand_valid, start):
    qx = query_x[start:]
    qo = query_obs[start:]
    cx = cand_x[:, start:]
    cv = cand_valid[:, start:]
    m = qo[None, :, :] & cv
    n = m.sum(axis=(1, 2))
    d = np.abs(cx - qx[None, :, :])
    num = np.where(m, d, 0.0).sum(axis=(1, 2), dtype=np.float64)
    mae = np.divide(num, n, out=np.full(len(cx), np.inf, dtype=np.float64), where=n > 0)
    denom = max(int(qo.sum()), 1)
    overlap = n.astype(np.float64) / float(denom)
    return mae, overlap


def _observed_state_scores(query_x, query_obs, cand_x, cand_valid, cfg):
    H = int(query_x.shape[0])
    specs = {
        "recent3": max(0, H - 3),
        "recent6": max(0, H - 6),
        "all": 0,
    }
    temp = max(float(cfg["retrieval"].get("observed_temperature", 0.75)), 1e-6)
    weights = cfg["retrieval"].get("observed_scope_weights", {"recent3": 0.65, "recent6": 0.25, "all": 0.10})

    maes, overlaps, sims = {}, {}, {}
    num = np.zeros(len(cand_x), dtype=np.float64)
    den = np.zeros(len(cand_x), dtype=np.float64)
    mae_num = np.zeros(len(cand_x), dtype=np.float64)
    mae_den = np.zeros(len(cand_x), dtype=np.float64)
    for name, start in specs.items():
        mae, ov = _scope_stats(query_x, query_obs, cand_x, cand_valid, start)
        finite = np.isfinite(mae)
        sim = np.zeros(len(cand_x), dtype=np.float64)
        sim[finite] = np.exp(-mae[finite] / temp)
        w = float(weights.get(name, 0.0))
        num += w * sim
        den += w * finite.astype(np.float64)
        mae_num += w * np.where(finite, mae, 0.0)
        mae_den += w * finite.astype(np.float64)
        maes[name] = mae
        overlaps[name] = ov
        sims[name] = sim

    combined_sim = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    combined_mae = np.divide(mae_num, mae_den, out=np.full_like(mae_num, np.inf), where=mae_den > 0)
    return combined_sim, combined_mae, maes, overlaps, sims


def retrieve_queries(cfg, adapter, ck, train, val, memory_npz, query_ids, device):
    mem = np.load(memory_npz)
    train_ids = mem["train_ids"].astype(np.int64)
    keys = mem["latent_keys"].astype(np.float32)
    mtod = mem["tod_last"]
    mdow = mem["dow_last"]
    pre_k = min(int(cfg["retrieval"]["prefilter_k"]), len(train_ids))
    top_k = min(int(cfg["retrieval"]["top_k"]), pre_k)

    names = ["recent3", "recent6", "all"]
    out = {k: [] for k in [
        "candidate_ids", "score", "latent_sim", "observed_mae", "context_score", "overlap_fraction",
        "observed_mae_recent3", "observed_mae_recent6", "observed_mae_all",
        "overlap_recent3", "overlap_recent6", "overlap_all",
    ]}
    lw = float(cfg["retrieval"]["latent_weight"])
    ow = float(cfg["retrieval"]["observed_weight"])
    cw = float(cfg["retrieval"]["context_weight"])
    minfrac = float(cfg["retrieval"]["require_candidate_valid_fraction"])

    for qi in query_ids:
        q = val.row(int(qi))
        _, pre = predict_raw(adapter, ck, q["x_obs"], q["tod_hist"], q["dow_hist"], device, return_prehead=True)
        qk = pooled_prehead(pre)[0]
        qk = qk / (np.linalg.norm(qk) + 1e-8)
        lat = keys @ qk
        ctx = _context_score(q["tod_hist"][-1], q["dow_hist"][-1], mtod, mdow, cfg)
        pre_score = lw * lat + cw * ctx
        cand_local = np.argpartition(-pre_score, pre_k - 1)[:pre_k]
        cand_local = cand_local[np.argsort(-pre_score[cand_local])]
        tids = train_ids[cand_local]

        cx = np.stack([np.asarray(train.x_full_norm[i], np.float32) for i in tids])
        cv = np.stack([train.row(int(i))["full_valid"] for i in tids])
        osim, omae, scope_mae, scope_overlap, _ = _observed_state_scores(
            q["x_obs"], q["history_observed"], cx, cv, cfg
        )
        score = lw * lat[cand_local] + ow * osim + cw * ctx[cand_local]
        # Retain the original all-history validity guard. v0.3 retains the
        # recent-state ranking emphasis without silently changing the memory support pool.
        score = np.where(scope_overlap["all"] >= minfrac, score, -np.inf)
        order = np.argsort(-score)[:top_k]

        out["candidate_ids"].append(tids[order])
        out["score"].append(score[order])
        out["latent_sim"].append(lat[cand_local][order])
        out["observed_mae"].append(omae[order])
        out["context_score"].append(ctx[cand_local][order])
        out["overlap_fraction"].append(scope_overlap["all"][order])
        for name in names:
            out[f"observed_mae_{name}"].append(scope_mae[name][order])
            out[f"overlap_{name}"].append(scope_overlap[name][order])

    return {
        "query_ids": np.asarray(query_ids, np.int64),
        **{k: np.asarray(v, np.int64 if k == "candidate_ids" else np.float32) for k, v in out.items()},
    }
