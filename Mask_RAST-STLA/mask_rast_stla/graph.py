"""Train-only statistical neighbourhoods; never call these physical road edges."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch


def graph_neighbours(cache, out, neighbours=8, sampled_rows=1200, force=False):
    cache=Path(cache);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    a=np.load(cache/'train_x_full_norm.npy',mmap_mode='r')
    n=int(a.shape[-1]); k=min(int(neighbours),max(n-1,1))
    ident=hashlib.sha256(f'{cache.resolve()}|{a.shape}|{k}|{sampled_rows}|{(cache / "train_x_full_norm.npy").stat().st_mtime_ns}'.encode()).hexdigest()[:20]
    meta=out.with_suffix('.json')
    if out.is_file() and meta.is_file() and not force:
        info=json.loads(meta.read_text())
        if info.get('fingerprint')==ident:
            z=np.load(out);return torch.from_numpy(z['neighbors'].astype(np.int64))
    idx=np.linspace(0,len(a)-1,min(int(sampled_rows),len(a)),dtype=np.int64)
    x=np.asarray(a[idx],np.float32).reshape(-1,n)
    x=np.nan_to_num(x,copy=False)
    x=(x-x.mean(0,keepdims=True))/(x.std(0,keepdims=True)+1e-5)
    sim=np.abs((x.T@x)/max(len(x),1));np.fill_diagonal(sim,-np.inf)
    if n==1: nn=np.zeros((1,1),dtype=np.int64)
    else:
        nn=np.argsort(sim,axis=1)[:,-k:][:,::-1].copy()
    np.savez_compressed(out,neighbors=nn)
    meta.write_text(json.dumps({'fingerprint':ident,'nodes':n,'neighbours':k,
       'source':'train_x_full_norm only','note':'statistical association, not directed traffic topology'},indent=2))
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
