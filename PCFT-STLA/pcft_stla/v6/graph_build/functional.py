"""Train-only observed traffic-change lag correlations. Not causal/physical edges."""
from __future__ import annotations
import numpy as np

FUNCTIONAL_FIELDS=('signed_lag_corr','abs_lag_corr','lag_scaled','support_scaled','split_stability','physical_edge_overlap')


def observed_train_series(ds,max_timestamps=8192):
    """Deduplicate overlapping TRAIN windows and never union artificial sensing views.

    At each timestamp, use values and mask from the earliest selected TRAIN view
    covering that time. Never use complete unseen history / val / test.
    """
    starts=ds.meta['window_start'].to_numpy(np.int64)[:len(ds)]
    order=np.argsort(starts,kind='stable');seen_windows=set();chosen=[]
    for row in order:
        w=int(starts[row])
        if w in seen_windows:continue
        seen_windows.add(w);chosen.append(int(row))
        if len(chosen)>max_timestamps:break
    if not chosen:raise RuntimeError('No unique TRAIN windows')
    lo=int(min(starts[chosen]));hi=min(int(max(starts[chosen]))+ds.x.shape[1],lo+max_timestamps)
    n=ds.x.shape[2];values=np.zeros((hi-lo,n),dtype=np.float32)
    masks=np.zeros((hi-lo,n),dtype=bool);assigned=np.zeros((hi-lo,),dtype=bool)
    ch=ds.target_idx
    for row in chosen:
        t=int(starts[row]);a=max(t,lo);b=min(t+ds.x.shape[1],hi)
        if b<=a:continue
        sl=slice(a-lo,b-lo);hs=slice(a-t,b-t)
        current=~assigned[sl]
        if np.any(current):
            ii=np.flatnonzero(current)+a-lo
            js=ii+lo-t
            values[ii]=np.asarray(ds.x[row,js,:,ch],dtype=np.float32)
            masks[ii]=np.asarray(ds.obs[row,js,:,ch],dtype=np.uint8).astype(bool)
            assigned[ii]=True
        if assigned.all():break
    return values,masks,{'unique_train_windows':len(seen_windows),'train_timestamps':int(assigned.sum()),
                         'retained_observations':int(masks.sum()),'first_timestamp':lo,
                         'last_timestamp_exclusive':hi,'deduplication':'one observed view per timestamp'}


def _corr(x,y,valid,min_pairs):
    n=int(valid.sum())
    if n<min_pairs:return None
    a=x[valid].astype(np.float64);b=y[valid].astype(np.float64)
    a-=a.mean();b-=b.mean();den=float(np.sqrt(np.dot(a,a)*np.dot(b,b)))
    return (float(np.dot(a,b)/den),n) if den>1e-9 else None


def build_functional(ds,physical_index,topk=8,max_lag=6,max_timestamps=8192,
                     shortlist=16,min_pairs=64,min_abs_corr=.10):
    values,obs,audit=observed_train_series(ds,max_timestamps)
    T,N=values.shape
    # Train changes must be valid at both adjacent observed instants.
    diff=np.zeros_like(values);dm=np.zeros_like(obs)
    diff[1:]=values[1:]-values[:-1];dm[1:]=obs[1:]&obs[:-1]
    diff[~dm]=0.
    # Coarse shortlist uses TRAIN-only zero-filled change vectors, then
    # every candidate is scored with exact pairwise-valid lagged correlation.
    length=np.sqrt((diff*diff).sum(0)).clip(1e-5)
    unit=diff/length[None]
    coarse=unit.T@unit
    np.fill_diagonal(coarse,0.)
    physical=set(zip(physical_index[0].tolist(),physical_index[1].tolist()))
    results=[]
    for dst in range(N):
        candidates=np.argpartition(np.abs(coarse[:,dst]),-min(shortlist,N-1))[-min(shortlist,N-1):]
        candidates=set(int(x) for x in candidates if x!=dst)
        candidates.update(src for src,d in physical if d==dst)
        scores=[]
        for src in candidates:
            best=None
            for lag in range(1,max_lag+1):
                corr=_corr(diff[:-lag,src],diff[lag:,dst],dm[:-lag,src]&dm[lag:,dst],min_pairs)
                if corr is None:continue
                rho,n=corr
                # Stability against distinct halves: avoid coincidental sign flips.
                half=max(3,(T-lag)//2)
                c1=_corr(diff[:-lag,src][:half],diff[lag:,dst][:half],
                         (dm[:-lag,src]&dm[lag:,dst])[:half],max(10,min_pairs//4))
                c2=_corr(diff[:-lag,src][half:],diff[lag:,dst][half:],
                         (dm[:-lag,src]&dm[lag:,dst])[half:],max(10,min_pairs//4))
                stability=(float(np.sign(c1[0])==np.sign(rho) and np.sign(c2[0])==np.sign(rho))
                           if c1 is not None and c2 is not None else 0.)
                if stability==0.0:continue
                score=abs(rho)*min(1.,n/400.)*(0.5+0.5*stability)
                if best is None or score>best[0]:best=(score,rho,lag,n,stability)
            if best is None or abs(best[1])<min_abs_corr:continue
            sc,rho,lag,n,stab=best
            attrs=[rho,abs(rho),lag/max_lag,min(n/400.,1.),stab,float((src,dst) in physical)]
            scores.append((sc,src,dst,attrs))
        scores.sort(key=lambda x:(-x[0],x[1]))
        results.extend(scores[:topk])
    if results:
        ei=np.asarray([(x[1],x[2]) for x in results],dtype=np.int64).T.copy()
        ea=np.asarray([x[3] for x in results],dtype=np.float32)
    else:
        # A graph lacking stable functional edges is valid: model must retain
        # a local-physical forecasting path and avoid inventing correlations.
        ei=np.empty((2,0),dtype=np.int64);ea=np.empty((0,len(FUNCTIONAL_FIELDS)),dtype=np.float32)
    if not np.isfinite(ea).all():raise RuntimeError('Nonfinite functional edge attributes')
    audit.update(functional_edges=int(ei.shape[1]),max_lag=max_lag,topk=topk,
                 min_pairs=min_pairs,min_abs_corr=min_abs_corr,
                 functional_schema=list(FUNCTIONAL_FIELDS),target=ds.target,
                 causal_claim=False,source='train_s40_observed_only')
    return ei,ea,audit
