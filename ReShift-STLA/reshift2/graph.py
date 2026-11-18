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
    identity=f'reshift2_poslag|{cache.resolve()}|{a.shape}|{k}|{sampled_rows}|{lags}|{(cache / "train_x_full_norm.npy").stat().st_mtime_ns}'
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
