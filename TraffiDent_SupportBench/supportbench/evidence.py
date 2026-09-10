from __future__ import annotations

from typing import Any
import math
import numpy as np


def _finite_or_none(x: float | None):
    if x is None: return None
    x=float(x)
    return x if math.isfinite(x) else None


def series_stats(x: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    x=np.asarray(x,dtype=np.float32).reshape(-1)
    m=np.isfinite(x) if mask is None else (np.asarray(mask,dtype=bool).reshape(-1)&np.isfinite(x))
    idx=np.flatnonzero(m)
    if not len(idx):
        return {'coverage':0.0,'count':0,'mean':None,'std':None,'first':None,'last':None,'trend':None,'min':None,'max':None}
    vals=x[idx]
    return {
        'coverage':float(len(idx)/len(x)),'count':int(len(idx)),'mean':_finite_or_none(vals.mean()),'std':_finite_or_none(vals.std()),
        'first':_finite_or_none(vals[0]),'last':_finite_or_none(vals[-1]),'trend':_finite_or_none(vals[-1]-vals[0]),
        'min':_finite_or_none(vals.min()),'max':_finite_or_none(vals.max()),
    }


def aggregate_neighbors(x: np.ndarray, obs: np.ndarray, ids: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x,np.float32); obs=np.asarray(obs,bool); ids=np.asarray(ids,np.int64)
    ids=ids[ids>=0]
    H=x.shape[0]
    if not len(ids): return np.full(H,np.nan,np.float32),np.zeros(H,bool)
    if weights is None:
        w=np.ones(len(ids),np.float32)
    else:
        ww=np.asarray(weights,np.float32).reshape(-1)[:len(np.asarray(weights).reshape(-1))]
        # Directed arrays keep valid neighbours first; trim to number of valid ids.
        w=ww[:len(ids)] if len(ww)>=len(ids) else np.ones(len(ids),np.float32)
    vals=x[:,ids]; valid=obs[:,ids]&np.isfinite(vals)
    out=np.full(H,np.nan,np.float32); out_valid=np.any(valid,axis=1)
    for t in np.flatnonzero(out_valid):
        wt=np.where(valid[t],w,0.0); den=float(wt.sum())
        if den<=0: wt=np.where(valid[t],1.0,0.0); den=float(wt.sum())
        out[t]=float((np.where(valid[t],vals[t],0.0)*wt).sum()/max(den,1e-8))
    return out,out_valid


def best_lead_lag_corr(leader: np.ndarray, follower: np.ndarray, leader_mask: np.ndarray, follower_mask: np.ndarray, max_lag: int = 3) -> dict[str,Any]:
    leader=np.asarray(leader,float); follower=np.asarray(follower,float); lm=np.asarray(leader_mask,bool); fm=np.asarray(follower_mask,bool)
    best=None
    for lag in range(0,int(max_lag)+1):
        if lag==0:
            a,b=leader,follower; ma,mb=lm,fm
        else:
            a,b=leader[:-lag],follower[lag:]; ma,mb=lm[:-lag],fm[lag:]
        m=ma&mb&np.isfinite(a)&np.isfinite(b)
        if int(m.sum())<3: continue
        aa=a[m]; bb=b[m]
        if float(np.std(aa))<1e-8 or float(np.std(bb))<1e-8: corr=0.0
        else: corr=float(np.corrcoef(aa,bb)[0,1])
        candidate={'lag_steps':int(lag),'corr':_finite_or_none(corr),'n_pairs':int(m.sum())}
        if best is None or abs(float(corr))>abs(float(best['corr'])): best=candidate
    return best or {'lag_steps':None,'corr':None,'n_pairs':0}


def select_anchor_nodes(bench, sample, max_anchors: int = 3) -> np.ndarray:
    miss=np.asarray(sample.artificial_mask,bool).mean(axis=0)
    obs=np.asarray(sample.observation_mask[...,bench.speed_idx],bool)
    degree=np.asarray(bench.graph_degree,dtype=float)
    # Prefer sensing-affected boundary/relational nodes: missingness is primary,
    # then availability of observed upstream/downstream context, then degree.
    rel_context=np.zeros(bench.N,float)
    for n in range(bench.N):
        ids=np.concatenate([bench.upstream_ids[n][bench.upstream_ids[n]>=0],bench.downstream_ids[n][bench.downstream_ids[n]>=0]])
        if len(ids): rel_context[n]=float(obs[:,ids].mean())
    score=miss.astype(float)*1000.0+rel_context*100.0+degree.astype(float)
    order=np.lexsort((np.arange(bench.N),-score))
    if np.any(miss>0): order=np.asarray([i for i in order if miss[i]>0],dtype=np.int64)
    if len(order)<max_anchors:
        extra=[i for i in np.argsort(-(rel_context*100.0+degree)) if i not in set(map(int,order))]
        order=np.asarray(list(order)+extra,dtype=np.int64)
    return order[:max(1,int(max_anchors))].astype(np.int64)


def _anchor_query(bench, sample, node: int, max_lag: int) -> dict[str,Any]:
    x=np.asarray(sample.history_full[...,bench.speed_idx],np.float32)
    obs=np.asarray(sample.observation_mask[...,bench.speed_idx],bool)
    art=np.asarray(sample.artificial_mask,bool)
    upids=bench.upstream_ids[node]; downids=bench.downstream_ids[node]
    up,upm=aggregate_neighbors(x,obs,upids,bench.upstream_weights[node]); down,downm=aggregate_neighbors(x,obs,downids,bench.downstream_weights[node])
    target=x[:,node]; tm=obs[:,node]
    return {
        'node_id':int(node),
        'road_direction':str(bench.nodes.iloc[int(node)].get('road_direction','')),
        'target_missing_fraction':float(art[:,node].mean()),
        'predictive_neighbor_ids':[int(i) for i in bench.dependency_ids[node] if int(i)>=0],
        'predictive_neighbor_weights':[float(w) for i,w in zip(bench.dependency_ids[node],bench.dependency_weights[node]) if int(i)>=0],
        'predictive_neighbor_lag_steps':[int(l) for i,l in zip(bench.dependency_ids[node],bench.dependency_lags[node]) if int(i)>=0],
        'predictive_neighbor_lag_minutes':[int(l) for i,l in zip(bench.dependency_ids[node],bench.dependency_lag_minutes[node]) if int(i)>=0],
        # Legacy alias kept for downstream readers; values are raw 5-min lag steps.
        'predictive_neighbor_lags':[int(l) for i,l in zip(bench.dependency_ids[node],bench.dependency_lags[node]) if int(i)>=0],
        'predictive_neighbor_corr':[float(c) for i,c in zip(bench.dependency_ids[node],bench.dependency_corr[node]) if int(i)>=0],
        'upstream_node_ids':[int(i) for i in upids if int(i)>=0],
        'downstream_node_ids':[int(i) for i in downids if int(i)>=0],
        'upstream_missing_fraction':float(art[:,upids[upids>=0]].mean()) if np.any(upids>=0) else None,
        'downstream_missing_fraction':float(art[:,downids[downids>=0]].mean()) if np.any(downids>=0) else None,
        'target_observed':series_stats(target,tm),
        'upstream_observed':series_stats(up,upm),
        'downstream_observed':series_stats(down,downm),
        'observed_upstream_to_target':best_lead_lag_corr(up,target,upm,tm,max_lag),
        'observed_target_to_downstream':best_lead_lag_corr(target,down,tm,downm,max_lag),
    }


def _anchor_candidate(bench, episode: dict, node: int, max_lag: int) -> dict[str,Any]:
    h=np.asarray(episode['history_speed'],np.float32); f=np.asarray(episode['future_speed'],np.float32)
    hm=np.ones_like(h,dtype=bool)
    upids=bench.upstream_ids[node]; downids=bench.downstream_ids[node]
    up,upm=aggregate_neighbors(h,hm,upids,bench.upstream_weights[node]); down,downm=aggregate_neighbors(h,hm,downids,bench.downstream_weights[node])
    target=h[:,node]; tm=np.isfinite(target)
    fut_target=f[:,node]
    return {
        'node_id':int(node),
        'target_history':series_stats(target,tm),
        'upstream_history':series_stats(up,upm),
        'downstream_history':series_stats(down,downm),
        'historical_upstream_to_target':best_lead_lag_corr(up,target,upm,tm,max_lag),
        'historical_target_to_downstream':best_lead_lag_corr(target,down,tm,downm,max_lag),
        'target_historical_future':series_stats(fut_target,np.isfinite(fut_target)),
    }


def _global_query(bench,sample) -> dict[str,Any]:
    x=np.asarray(sample.history_full[...,bench.speed_idx],np.float32); obs=np.asarray(sample.observation_mask[...,bench.speed_idx],bool)
    step=[]; stepmask=[]
    for t in range(bench.H):
        vals=x[t][obs[t]]
        step.append(float(vals.mean()) if len(vals) else np.nan); stepmask.append(bool(len(vals)))
    return {
        'observed_speed':series_stats(np.asarray(step,float),np.asarray(stepmask,bool)),
        'observed_fraction':float(obs.mean()),
        'masked_node_fraction':float(np.asarray(sample.artificial_mask,bool).any(axis=0).mean()),
    }


def _global_candidate(episode: dict) -> dict[str,Any]:
    h=np.asarray(episode['history_speed'],np.float32); f=np.asarray(episode['future_speed'],np.float32)
    return {
        'history_speed':series_stats(h.mean(axis=1),np.ones(h.shape[0],bool)),
        'historical_future_speed':series_stats(f.mean(axis=1),np.ones(f.shape[0],bool)),
    }


def build_pair_evidence(bench, sample, episode_id: int, max_anchors: int = 3, max_lag: int = 3) -> dict[str,Any]:
    anchors=select_anchor_nodes(bench,sample,max_anchors); episode=bench.memory_episode(int(episode_id))
    query_anchors=[_anchor_query(bench,sample,int(n),max_lag) for n in anchors]
    cand_anchors=[_anchor_candidate(bench,episode,int(n),max_lag) for n in anchors]
    # Machine-verifiable relation summaries used by grounded explanation checks.
    propositions=[]
    for q,c in zip(query_anchors,cand_anchors):
        qtr=q['target_observed']['trend']; ctr=c['target_history']['trend']; pc=c['historical_upstream_to_target']['corr']
        propositions.append({
            'node_id':q['node_id'],
            'target_is_substantially_missing':bool(q['target_missing_fraction']>=0.5),
            'upstream_is_more_missing_than_target':None if q['upstream_missing_fraction'] is None else bool(q['upstream_missing_fraction']>q['target_missing_fraction']),
            'candidate_target_trend_same_sign':None if qtr is None or ctr is None else bool(float(qtr)*float(ctr)>=0),
            'candidate_upstream_to_target_positive_corr':None if pc is None else bool(float(pc)>0),
        })
    context=bench.predictive_context_metrics(sample)
    return {
        'anchor_node_ids':[int(x) for x in anchors],
        'query_predictive_context':{
            'predictive_context_loss':float(context['network_mean_context_loss']),
            'network_mean_context_loss':float(context['network_mean_context_loss']),
            'support_dependent_fraction':float(context['support_dependent_fraction']),
            'legacy_context_loss_quantile':float(context['context_loss_quantile_value']),
            'aggregate_context_loss':float(context['aggregate_context_loss']),
        },
        'query_global':_global_query(bench,sample),
        'candidate_global':_global_candidate(episode),
        'query_anchors':query_anchors,
        'candidate_anchors':cand_anchors,
        'verifiable_propositions':propositions,
    }
