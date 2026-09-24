from __future__ import annotations
import numpy as np


def _softmax(x, temp):
    z = np.asarray(x, np.float64) / max(float(temp), 1e-6)
    z = z - np.max(z)
    e = np.exp(z)
    return (e / max(float(e.sum()), 1e-12)).astype(np.float32)


def _sigmoid(x):
    x = float(np.clip(x, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-x))


def fuse_residual(base_pred, direct_mask, memory_arrays, retrieved, cfg, data_std):
    if retrieved is None or len(retrieved["local"]) == 0:
        return np.asarray(base_pred, np.float32).copy(), {
            "used": False, "gate": 0.0, "alpha_eff": 0.0, "correction_abs_mean": 0.0,
            "mean_latent_sim": float("nan"), "residual_disagreement": float("nan"),
        }, None
    loc = retrieved["local"]
    residual = memory_arrays["residual"][loc].astype(np.float32)
    valid = memory_arrays["valid"][loc].astype(bool)
    w = _softmax(retrieved["score"], cfg["softmax_temperature"])
    H, N = residual.shape[1], residual.shape[2]
    numer = np.zeros((H,N), np.float64); denom = np.zeros((H,N), np.float64)
    for j in range(len(loc)):
        m = valid[j] & np.isfinite(residual[j])
        numer[m] += float(w[j]) * residual[j][m]
        denom[m] += float(w[j])
    correction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom>0).astype(np.float32)

    # Evidence disagreement: weighted cell-wise std, averaged over cells with >=2 valid supports.
    var_num = np.zeros((H,N), np.float64); var_den = np.zeros((H,N), np.float64)
    for j in range(len(loc)):
        m = valid[j] & (denom > 0)
        d = residual[j] - correction
        var_num[m] += float(w[j]) * (d[m] ** 2)
        var_den[m] += float(w[j])
    var = np.divide(var_num, var_den, out=np.zeros_like(var_num), where=var_den>0)
    disagreement = float(np.sqrt(np.maximum(var,0.0)).mean())

    use_gate = bool(cfg.get("use_confidence_gate", True))
    if use_gate:
        ms = float(np.mean(retrieved["latent_sim"]))
        sim_gate = _sigmoid((ms - float(cfg["similarity_gate_center"])) / max(float(cfg["similarity_gate_scale"]),1e-6))
        dis_gate = float(np.exp(-disagreement / max(float(cfg["disagreement_scale_raw"]),1e-6)))
        gate = float(np.clip(sim_gate * dis_gate, 0.0, 1.0))
    else:
        gate = 1.0
    alpha_eff = float(cfg["alpha"]) * gate
    clip = float(cfg["correction_clip_sigma"]) * float(data_std)
    if clip > 0:
        correction = np.clip(correction, -clip, clip)
    scope = str(cfg.get("correction_scope", "direct_mask"))
    scope_mask = np.ones(N, bool) if scope == "all" else np.asarray(direct_mask, bool)
    applied = correction * scope_mask[None,:].astype(np.float32) * alpha_eff
    pred = np.asarray(base_pred, np.float32) + applied
    info = {
        "used": bool(alpha_eff > 0), "gate": gate, "alpha_eff": alpha_eff,
        "correction_abs_mean": float(np.mean(np.abs(applied))),
        "mean_latent_sim": float(np.mean(retrieved["latent_sim"])),
        "max_latent_sim": float(np.max(retrieved["latent_sim"])),
        "mean_age": float(np.mean(retrieved["age"])),
        "residual_disagreement": disagreement,
    }
    return pred, info, {"weights": w, "correction": correction}


def oracle_best_candidate(base_pred, y, current_valid, direct_mask, memory_arrays, retrieved, cfg, data_std):
    """Diagnostic only. Uses current labels after prediction; never feeds the production RAG path."""
    if retrieved is None or len(retrieved["local"]) == 0:
        return float("nan"), 0.0
    base = np.asarray(base_pred,np.float32); yy=np.asarray(y,np.float32); cv=np.asarray(current_valid,bool)
    dm=np.asarray(direct_mask,bool)[None,:]
    m=cv & dm & np.isfinite(yy)
    if not np.any(m): return float("nan"), 0.0
    best = float(np.abs(base-yy)[m].mean()); best_gain = 0.0
    clip = float(cfg["correction_clip_sigma"]) * float(data_std)
    for li in retrieved["local"]:
        r = memory_arrays["residual"][li].astype(np.float32)
        valid = memory_arrays["valid"][li]
        corr = np.where(valid, r, 0.0)
        if clip > 0: corr = np.clip(corr,-clip,clip)
        scope_mask = np.ones(base.shape[1],bool) if str(cfg.get("correction_scope","direct_mask"))=="all" else np.asarray(direct_mask,bool)
        p = base + float(cfg["alpha"]) * corr * scope_mask[None,:]
        mae = float(np.abs(p-yy)[m].mean())
        if mae < best:
            best = mae
    base_mae=float(np.abs(base-yy)[m].mean())
    if base_mae>0: best_gain=(base_mae-best)/base_mae*100.0
    return best,best_gain
