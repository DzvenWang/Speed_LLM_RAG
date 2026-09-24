from __future__ import annotations
import numpy as np


def _tod_similarity(q, c, slots_per_day):
    period = max(int(slots_per_day), 2)
    dt = np.abs(np.asarray(c, np.float32) - float(q))
    dt = np.minimum(dt, period - dt)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * dt / period))


def _observed_similarity(qx, qobs, cx, cobs, scale, min_overlap_fraction):
    overlap = np.asarray(qobs, bool) & np.asarray(cobs, bool) & np.isfinite(qx) & np.isfinite(cx)
    qn = max(int(np.asarray(qobs, bool).sum()), 1)
    frac = float(overlap.sum()) / float(qn)
    if frac < float(min_overlap_fraction) or not np.any(overlap):
        return 0.0, frac
    mae = float(np.mean(np.abs(np.asarray(qx, np.float32)[overlap] - np.asarray(cx, np.float32)[overlap])))
    sim = float(np.exp(-mae / max(float(scale), 1e-6)))
    return sim, frac


def _node_observed_similarity(qx, qobs, cx, cobs, node, scale, min_steps):
    qx=np.asarray(qx,np.float32); cx=np.asarray(cx,np.float32)
    qobs=np.asarray(qobs,bool); cobs=np.asarray(cobs,bool)
    n=int(node)
    overlap=qobs[:,n] & cobs[:,n] & np.isfinite(qx[:,n]) & np.isfinite(cx[:,n])
    cnt=int(overlap.sum())
    qcnt=max(int((qobs[:,n] & np.isfinite(qx[:,n])).sum()),1)
    frac=float(cnt)/float(qcnt)
    if cnt < int(min_steps):
        return 0.0, frac
    mae=float(np.mean(np.abs(qx[overlap,n]-cx[overlap,n])))
    return float(np.exp(-mae/max(float(scale),1e-6))), frac


def _complementarity(q_missing, cobs):
    qm = np.asarray(q_missing, bool)
    denom = int(qm.sum())
    if denom <= 0:
        return 0.0
    return float((qm & np.asarray(cobs, bool)).sum()) / float(denom)


def _node_complementarity(q_missing, cobs, node):
    qm=np.asarray(q_missing,bool)[:,int(node)]
    denom=int(qm.sum())
    if denom<=0:
        return 0.0
    return float((qm & np.asarray(cobs,bool)[:,int(node)]).sum())/float(denom)


def retrieve(memory, qkey, qorigin, qtod, qdow, qx, qobs, q_missing, cfg, top_k=None):
    """Global causal retrieval.

    `top_k` can override cfg[top_k]. v3.2 uses this to expose a larger ranked
    candidate pool for opportunity diagnostics while retaining Top-8 global
    completion as an exact v3.1 control.
    """
    arrays = memory.arrays()
    if arrays is None or len(arrays['index']) == 0:
        return None
    rows = memory.rows()
    keys = arrays['keys']
    latent = keys @ qkey.astype(np.float32)
    tod = _tod_similarity(qtod, arrays['tod'], cfg['slots_per_day'])
    dow = (arrays['dow'] == int(qdow)).astype(np.float32)
    context = float(cfg['tod_weight']) * tod + float(cfg['dow_weight']) * dow
    age = np.maximum(0, int(qorigin) - arrays['origin']).astype(np.float32)
    recency = np.exp(-age / max(float(cfg['recency_tau_steps']), 1.0))

    pre_score = (
        float(cfg['latent_weight']) * latent
        + float(cfg['recency_weight']) * recency
        + float(cfg['context_weight']) * context
    )
    m = min(int(cfg['prefilter_k']), len(pre_score))
    pre = np.argsort(-pre_score)[:m]

    obs_sim = np.zeros(m, np.float32)
    overlap_frac = np.zeros(m, np.float32)
    comp = np.zeros(m, np.float32)
    for j, li in enumerate(pre):
        r = rows[int(li)]
        s, f = _observed_similarity(
            qx, qobs, r.x_obs, r.history_observed,
            cfg['observed_match_scale'], cfg['min_overlap_fraction']
        )
        obs_sim[j] = s
        overlap_frac[j] = f
        comp[j] = _complementarity(q_missing, r.history_observed)

    score = (
        float(cfg['latent_weight']) * latent[pre]
        + float(cfg['observed_match_weight']) * obs_sim
        + float(cfg['complementarity_weight']) * comp
        + float(cfg['recency_weight']) * recency[pre]
        + float(cfg['context_weight']) * context[pre]
    )
    want=int(cfg['top_k']) if top_k is None else int(top_k)
    k = min(want, len(score))
    order0 = np.argsort(-score)[:k]
    chosen = pre[order0]
    return {
        'local': chosen,
        'case_index': arrays['index'][chosen],
        'score': score[order0].astype(np.float32),
        'latent_sim': latent[chosen].astype(np.float32),
        'observed_match': obs_sim[order0].astype(np.float32),
        'overlap_fraction': overlap_frac[order0].astype(np.float32),
        'complementarity': comp[order0].astype(np.float32),
        'recency': recency[chosen].astype(np.float32),
        'context': context[chosen].astype(np.float32),
        'age': age[chosen].astype(np.float32),
    }


def prefix_retrieval(retrieved, k):
    if retrieved is None:
        return None
    n=min(int(k),len(retrieved['case_index']))
    out={}
    for key,val in retrieved.items():
        if isinstance(val,np.ndarray) and val.ndim>=1 and len(val)==len(retrieved['case_index']):
            out[key]=val[:n].copy()
        else:
            out[key]=val
    return out


def nodewise_retrieve(memory, global_pool, qx, qobs, q_missing, cfg, ccfg, rng=None, random_support=False):
    """Re-rank a causal global candidate pool separately for each missing node.

    Similarity/complementarity matrices are vectorized across candidates and nodes;
    only the final small Top-K selection loops over missing nodes. This keeps v3.2
    practical on Net-06 (hundreds of sensors, thousands of online queries).
    """
    if global_pool is None or len(global_pool['case_index'])==0:
        return None
    rows=memory.rows(); qx=np.asarray(qx,np.float32); qobs=np.asarray(qobs,bool); qm=np.asarray(q_missing,bool)
    miss_nodes=np.where(np.any(qm,axis=0))[0]
    if len(miss_nodes)==0:
        return {'nodes':{},'pool_case_index':global_pool['case_index'],'pool_size':len(global_pool['case_index'])}
    rng=np.random.default_rng(0) if rng is None else rng
    node_cfg=cfg.get('nodewise',{}); topk=int(node_cfg.get('top_k_per_node',4))
    local_scale=max(float(node_cfg.get('local_observed_match_scale',cfg.get('observed_match_scale',0.5))),1e-6)
    min_steps=int(node_cfg.get('min_local_overlap_steps',1))

    local=np.asarray(global_pool['local'],np.int64); K=len(local)
    cx=np.stack([np.asarray(rows[int(li)].x_obs,np.float32) for li in local],axis=0)       # K,T,N
    cobs=np.stack([np.asarray(rows[int(li)].history_observed,bool) for li in local],axis=0)
    finite=np.isfinite(cx) & np.isfinite(qx)[None,:,:]
    overlap=cobs & qobs[None,:,:] & finite
    counts=overlap.sum(axis=1)                                                           # K,N
    qcounts=np.maximum((qobs & np.isfinite(qx)).sum(axis=0),1)[None,:]
    local_frac=counts/np.maximum(qcounts,1)
    absdiff=np.abs(cx-qx[None,:,:])
    mae=np.sum(np.where(overlap,absdiff,0.0),axis=1)/np.maximum(counts,1)
    local_sim=np.exp(-mae/local_scale).astype(np.float32)
    local_sim[counts<min_steps]=0.0

    miss_denom=np.maximum(qm.sum(axis=0),1)[None,:]
    node_comp=((cobs & qm[None,:,:]).sum(axis=1)/miss_denom).astype(np.float32)           # K,N
    node_comp[:,qm.sum(axis=0)==0]=0.0

    latent=np.nan_to_num(np.asarray(global_pool['latent_sim'],np.float32),nan=0.0)[:,None]
    gobs=np.nan_to_num(np.asarray(global_pool['observed_match'],np.float32),nan=0.0)[:,None]
    rec=np.nan_to_num(np.asarray(global_pool['recency'],np.float32),nan=0.0)[:,None]
    ctx=np.nan_to_num(np.asarray(global_pool['context'],np.float32),nan=0.0)[:,None]
    scores=(float(node_cfg.get('latent_weight',0.15))*latent
            +float(node_cfg.get('global_observed_match_weight',0.10))*gobs
            +float(node_cfg.get('local_observed_match_weight',0.30))*local_sim
            +float(node_cfg.get('complementarity_weight',0.25))*node_comp
            +float(node_cfg.get('recency_weight',0.15))*rec
            +float(node_cfg.get('context_weight',0.05))*ctx)

    result={}
    for n in miss_nodes:
        eligible=np.where(node_comp[:,int(n)]>0.0)[0]
        if len(eligible)==0:
            continue
        take=min(topk,len(eligible))
        if random_support:
            choose=rng.choice(eligible,size=take,replace=False)
            weights=np.full(take,1.0/max(take,1),np.float32)
        else:
            ss=scores[eligible,int(n)]
            choose=eligible[np.argsort(-ss)[:take]]
            cs=scores[choose,int(n)].astype(np.float32)
            temp=max(float(ccfg.get('score_temperature',0.15)),1e-6)
            z=(cs-cs.max())/temp; e=np.exp(np.clip(z,-40,40)); weights=(e/max(float(e.sum()),1e-12)).astype(np.float32)
        result[int(n)]={
            'pool_rank':choose.astype(np.int16),
            'local':local[choose].astype(np.int64),
            'case_index':np.asarray([rows[int(local[j])].index for j in choose],np.int64),
            'score':scores[choose,int(n)].astype(np.float32),
            'local_observed_match':local_sim[choose,int(n)].astype(np.float32),
            'local_overlap_fraction':local_frac[choose,int(n)].astype(np.float32),
            'node_complementarity':node_comp[choose,int(n)].astype(np.float32),
            'weights':weights,
        }
    return {'nodes':result,'pool_case_index':global_pool['case_index'],'pool_size':len(global_pool['case_index'])}


def random_retrieve(memory, qorigin, k, rng):
    rows = memory.rows()
    if not rows:
        return None
    n = min(int(k), len(rows))
    sel = rng.choice(np.arange(len(rows)), size=n, replace=False)
    idx = np.asarray([rows[int(i)].index for i in sel], np.int64)
    age = np.asarray([max(0, int(qorigin) - int(rows[int(i)].origin)) for i in sel], np.float32)
    return {
        'local': np.asarray(sel, np.int64),
        'case_index': idx,
        'score': np.ones(n, np.float32),
        'latent_sim': np.full(n, np.nan, np.float32),
        'observed_match': np.full(n, np.nan, np.float32),
        'overlap_fraction': np.full(n, np.nan, np.float32),
        'complementarity': np.full(n, np.nan, np.float32),
        'recency': np.full(n, np.nan, np.float32),
        'context': np.full(n, np.nan, np.float32),
        'age': age,
    }
