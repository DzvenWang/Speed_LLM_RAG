from __future__ import annotations
from pathlib import Path
import numpy as np

EPS = 1e-8


def history_features(x_obs: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Per-node causal history features from incomplete history only.

    x_obs: [..., H, N] normalized input with benchmark placeholders.
    observed: same shape boolean observation mask.
    returns [..., N, 6]: mean, std, last-observed, linear slope,
    last-minus-mean, observed coverage. No hidden query value is read.
    """
    x = np.asarray(x_obs, np.float32)
    m = np.asarray(observed, bool)
    H = x.shape[-2]
    w = m.astype(np.float32)
    cnt = w.sum(axis=-2)
    den = np.maximum(cnt, 1.0)
    mean = (x * w).sum(axis=-2) / den
    var = (((x - mean[..., None, :]) ** 2) * w).sum(axis=-2) / den
    std = np.sqrt(np.maximum(var, 0.0))

    rev = m[..., ::-1, :]
    last_from_end = np.argmax(rev, axis=-2)
    has = cnt > 0
    last_idx = (H - 1 - last_from_end).astype(np.int64)
    gather_idx = np.expand_dims(last_idx, axis=-2)
    last = np.take_along_axis(x, gather_idx, axis=-2).squeeze(-2)
    last = np.where(has, last, 0.0)

    t = np.arange(H, dtype=np.float32)
    shape = (1,) * (x.ndim - 2) + (H, 1)
    tt = t.reshape(shape)
    sw = cnt
    st = (w * tt).sum(axis=-2)
    stt = (w * tt * tt).sum(axis=-2)
    sx = (w * x).sum(axis=-2)
    stx = (w * tt * x).sum(axis=-2)
    cov = stx - st * sx / np.maximum(sw, 1.0)
    tv = stt - st * st / np.maximum(sw, 1.0)
    slope = np.where((sw >= 2) & (tv > EPS), cov / np.maximum(tv, EPS), 0.0)
    coverage = cnt / float(H)
    out = np.stack([mean, std, last, slope, last - mean, coverage], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_pca_from_memmap(prehead_path: Path, out_dim: int, max_vectors: int, seed: int):
    x = np.load(prehead_path, mmap_mode='r')  # [Q,N,D]
    Q,N,D = x.shape
    total = Q * N
    rng = np.random.default_rng(int(seed))
    take = min(int(max_vectors), total)
    idx = rng.choice(total, size=take, replace=False) if take < total else np.arange(total)
    q = idx // N; n = idx % N
    sample = np.asarray(x[q,n], np.float32)
    mean = sample.mean(axis=0)
    z = sample - mean
    cov = (z.T @ z) / max(len(z)-1,1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:min(int(out_dim),D)]
    comp = vecs[:,order].T.astype(np.float32)
    explained = vals[order] / max(float(np.maximum(vals,0).sum()), EPS)
    return mean.astype(np.float32), comp, explained.astype(np.float32)


def project_prehead(prehead: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    x = np.asarray(prehead, np.float32)
    return np.einsum('...d,kd->...k', x - mean, comp, optimize=True).astype(np.float32)


def pooled_key(node_latent: np.ndarray) -> np.ndarray:
    z = np.asarray(node_latent, np.float32).mean(axis=1)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    return z / np.maximum(norm, EPS)


def causal_topk(query_keys, memory_keys, query_start, memory_start, history, horizon,
                k=16, temperature=0.15, chunk=256):
    """Cosine Top-K with strict historical episode completion before query start."""
    qk = np.asarray(query_keys, np.float32)
    mk = np.asarray(memory_keys, np.float32)
    qs = np.asarray(query_start, np.int64)
    ms = np.asarray(memory_start, np.int64)
    end = ms + int(history) + int(horizon)
    Q = len(qk); K = int(k)
    out_i = np.full((Q,K), -1, np.int32)
    out_s = np.full((Q,K), -np.inf, np.float32)
    out_w = np.zeros((Q,K), np.float32)
    for a in range(0,Q,int(chunk)):
        b=min(a+int(chunk),Q)
        sim=qk[a:b] @ mk.T
        valid=end[None,:] < qs[a:b,None]
        sim=np.where(valid,sim,-np.inf)
        kk=min(K,sim.shape[1])
        part=np.argpartition(sim,-kk,axis=1)[:,-kk:]
        vals=np.take_along_axis(sim,part,axis=1)
        ord_=np.argsort(vals,axis=1)[:,::-1]
        inds=np.take_along_axis(part,ord_,axis=1)
        vals=np.take_along_axis(vals,ord_,axis=1)
        finite=np.isfinite(vals)
        inds=np.where(finite,inds,-1)
        safe=np.where(finite,vals,-1e9)
        mx=safe.max(axis=1,keepdims=True)
        ex=np.where(finite,np.exp((safe-mx)/max(float(temperature),1e-6)),0.0)
        w=ex/np.maximum(ex.sum(axis=1,keepdims=True),EPS)
        out_i[a:b,:kk]=inds.astype(np.int32)
        out_s[a:b,:kk]=np.where(finite,vals,-np.inf).astype(np.float32)
        out_w[a:b,:kk]=w.astype(np.float32)
    return out_i,out_s,out_w


def causal_random_k(query_start, memory_start, history, horizon, k, seed):
    """Causality-preserving random support control; never draws future memory."""
    qs=np.asarray(query_start,np.int64); ms=np.asarray(memory_start,np.int64)
    end=ms+int(history)+int(horizon); K=int(k); rng=np.random.default_rng(int(seed))
    out=np.full((len(qs),K),-1,np.int32); w=np.zeros((len(qs),K),np.float32)
    for i,t in enumerate(qs):
        elig=np.flatnonzero(end<t)
        if len(elig)==0: continue
        take=min(K,len(elig)); pick=rng.choice(elig,size=take,replace=False)
        out[i,:take]=pick.astype(np.int32); w[i,:take]=1.0/float(take)
    return out,w


def aggregate_support(indices, weights, mem_hist_feat, mem_future_evo, mem_future_mask):
    """Aggregate candidate evidence with per-node/per-horizon observability masks."""
    idx=np.asarray(indices,np.int64); ww=np.asarray(weights,np.float32)
    Q,K=idx.shape; N=mem_hist_feat.shape[1]; F=mem_hist_feat.shape[2]; P=mem_future_evo.shape[1]
    hist=np.zeros((Q,N,F),np.float32)
    evo=np.zeros((Q,P,N),np.float32)
    cov=np.zeros((Q,P,N),np.float32)
    for q in range(Q):
        ids=idx[q]; ws=ww[q]; ok=ids>=0
        if not np.any(ok): continue
        ids=ids[ok]; ws=ws[ok]
        hf=np.asarray(mem_hist_feat[ids],np.float32)  # K,N,F
        node_valid=hf[...,5] > 0
        hn=(ws[:,None,None]*hf*node_valid[...,None]).sum(axis=0)
        hd=(ws[:,None]*node_valid).sum(axis=0)
        hist[q]=hn/np.maximum(hd[:,None],EPS)
        fe=np.asarray(mem_future_evo[ids],np.float32)
        fm=np.asarray(mem_future_mask[ids],np.float32)
        num=(ws[:,None,None]*fe*fm).sum(axis=0)
        den=(ws[:,None,None]*fm).sum(axis=0)
        evo[q]=num/np.maximum(den,EPS)
        cov[q]=np.clip(den,0.0,1.0)
    return hist,evo,cov
