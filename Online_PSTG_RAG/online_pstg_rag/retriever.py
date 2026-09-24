from __future__ import annotations
import numpy as np


def _tod_similarity(q, c, slots_per_day):
    period = max(int(slots_per_day), 2)
    dt = np.abs(np.asarray(c, np.float32) - float(q))
    dt = np.minimum(dt, period - dt)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * dt / period))


def retrieve(memory_arrays, qkey, qorigin, qtod, qdow, cfg):
    if memory_arrays is None or len(memory_arrays["index"]) == 0:
        return None
    keys = memory_arrays["keys"]
    latent = keys @ qkey.astype(np.float32)
    tod = _tod_similarity(qtod, memory_arrays["tod"], cfg["slots_per_day"])
    dow = (memory_arrays["dow"] == int(qdow)).astype(np.float32)
    age = np.maximum(0, int(qorigin) - memory_arrays["origin"]).astype(np.float32)
    tau = max(float(cfg["recency_tau_steps"]), 1.0)
    recency = np.exp(-age / tau)
    context = float(cfg["tod_weight"]) * tod + float(cfg["dow_weight"]) * dow
    score = float(cfg["latent_weight"]) * latent + float(cfg["context_weight"]) * context + float(cfg["recency_weight"]) * recency
    k = min(int(cfg["top_k"]), len(score))
    order = np.argsort(-score)[:k]
    return {
        "local": order,
        "case_index": memory_arrays["index"][order],
        "score": score[order].astype(np.float32),
        "latent_sim": latent[order].astype(np.float32),
        "recency": recency[order].astype(np.float32),
        "context": context[order].astype(np.float32),
        "age": age[order].astype(np.float32),
    }
