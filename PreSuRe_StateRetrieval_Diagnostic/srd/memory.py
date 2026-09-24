from __future__ import annotations
from pathlib import Path
import numpy as np
from .evaluate import predict_raw, pooled_prehead
from .utils import deterministic_ids, log


def build_memory(cfg, adapter, ck, train, device, out_path, max_windows=None):
    max_windows = int(max_windows or cfg["memory"]["max_train_windows"])
    ids = deterministic_ids(len(train), max_windows, cfg["experiment"]["seed"])
    bs = int(cfg["memory"].get("build_batch_size", 32)); keys=[]; tod_last=[]; dow_last=[]; coverage=[]
    use_full = bool(cfg["memory"].get("use_complete_state_for_memory_key", True))
    for s in range(0, len(ids), bs):
        b = ids[s:s+bs]
        x = np.stack([np.asarray(train.x_full_norm[i] if use_full else train.x_obs_norm[i], np.float32) for i in b])
        tod = np.stack([np.asarray(train.tod_hist[i], np.int64) for i in b]); dow = np.stack([np.asarray(train.dow_hist[i], np.int64) for i in b])
        _, pre = predict_raw(adapter, ck, x, tod, dow, device, return_prehead=True)
        keys.append(pooled_prehead(pre))
        for i in b:
            r=train.row(int(i)); tod_last.append(int(r["tod_hist"][-1])); dow_last.append(int(r["dow_hist"][-1])); coverage.append(float(r["full_valid"].mean()))
        log(f"MEMORY keys {min(s+len(b),len(ids))}/{len(ids)}")
    keys=np.concatenate(keys,axis=0).astype(np.float32)
    norms=np.linalg.norm(keys,axis=1,keepdims=True)+1e-8; keys=keys/norms
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p, train_ids=ids, latent_keys=keys, tod_last=np.asarray(tod_last,np.int16), dow_last=np.asarray(dow_last,np.int8), coverage=np.asarray(coverage,np.float32))
    return p
