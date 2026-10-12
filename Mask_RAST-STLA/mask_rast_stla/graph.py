"""Train-only statistical neighbourhoods; never call these physical road edges."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch


def graph_neighbours(cache, out, neighbours=8, sampled_rows=1200, force=False, lag_steps=(0,1,2,3)):
    """Train-only positive, lag-aware statistical support, not physical topology.

    Signed positive associations are retained; negative correlations do not
    become strong edges just because their absolute magnitude is large.
    """
    cache=Path(cache);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    a=np.load(cache/'train_x_full_norm.npy',mmap_mode='r')
    n=int(a.shape[-1]);k=min(int(neighbours),max(n-1,1))
    lags=tuple(sorted(set(int(t) for t in lag_steps)))
    if not lags or min(lags)<0 or max(lags)>=int(a.shape[1]):
        raise ValueError('lag_steps must be nonnegative and shorter than history')
    identity=f'v23_poslag|{cache.resolve()}|{a.shape}|{k}|{sampled_rows}|{lags}|{(cache / "train_x_full_norm.npy").stat().st_mtime_ns}'
    ident=hashlib.sha256(identity.encode()).hexdigest()[:20]
    meta=out.with_suffix('.json')
    if out.is_file() and meta.is_file() and not force:
        info=json.loads(meta.read_text())
        if info.get('fingerprint')==ident:
            with np.load(out,allow_pickle=False) as z:return torch.from_numpy(z['neighbors'].astype(np.int64))
    idx=np.linspace(0,len(a)-1,min(int(sampled_rows),len(a)),dtype=np.int64)
    x=np.nan_to_num(np.array(a[idx],dtype=np.float32,copy=True))
    sim=np.zeros((n,n),np.float64)
    weights=0.
    for lag in lags:
        left=(x[:,:-lag,:] if lag else x).reshape(-1,n)
        right=(x[:,lag:,:] if lag else x).reshape(-1,n)
        # Normalize each graph covariance view with its own train statistics.
        left=(left-left.mean(0))/(left.std(0)+1e-5)
        right=(right-right.mean(0))/(right.std(0)+1e-5)
        signed=(left.T@right)/max(len(left),1)
        weight=1./(1.+lag)
        # Directionality is not asserted: symmetrize forward/backward lagged
        # statistics before feeding an undirected region grouping.
        sim+=weight*np.maximum((signed+signed.T)*0.5,0.)
        weights+=weight
    sim/=max(weights,1e-8)
    np.fill_diagonal(sim,-np.inf)
    if n==1:nn=np.zeros((1,1),dtype=np.int64)
    else:nn=np.argsort(sim,axis=1,kind='stable')[:,-k:][:,::-1].copy()
    np.savez_compressed(out,neighbors=nn,graph_score=sim.astype(np.float32),lags=np.asarray(lags,np.int16))
    meta.write_text(json.dumps({'fingerprint':ident,'nodes':n,'neighbours':k,'lags':list(lags),
       'source':'train_x_full_norm only','note':'positive lagged statistical association, not directed road topology'},indent=2))
    return torch.from_numpy(nn)


def assign_connected_regions(neighbors, target_size=32):
    """Deterministic undirected connectivity groups over TRAIN-only statistical graph.

    An isolated node starts its own group; no fixed Net-06 node indices.
    """
    a=np.asarray(neighbors.cpu() if isinstance(neighbors,torch.Tensor) else neighbors,dtype=np.int64)
    n=int(a.shape[0]);target_size=max(1,int(target_size))
    adj=[set() for _ in range(n)]
    for i in range(n):
        for j in a[i]:
            j=int(j)
            if 0<=j<n and i!=j:adj[i].add(j);adj[j].add(i)
    assignment=np.full(n,-1,dtype=np.int64);r=0
    for origin in range(n):
        if assignment[origin]>=0:continue
        frontier=[origin];assignment[origin]=r;count=1
        for node in frontier:
            if count>=target_size:break
            for nxt in sorted(adj[node]):
                if assignment[nxt]<0:
                    assignment[nxt]=r;frontier.append(nxt);count+=1
                    if count>=target_size:break
        r+=1
    return torch.from_numpy(assignment)
