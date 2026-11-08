"""Visible-input-only retrieval with four shared, time-legal historical anchors.

No other sensing view participates in a query's retrieval. The shared anchor
identities are selected from time metadata only, but ranked with THIS view's
visible measurements. The remaining positions use mask-conditioned similarity.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

class VisibleCandidateRetriever:
    def __init__(self,cache,limit=None,pool_size=128,shared=4):
        root=Path(cache)
        from ..data import SupportSplit
        ds=SupportSplit(root,'train',limit=limit)
        self.starts=np.asarray(ds.window_starts[:len(ds)],np.int64)
        if not np.all(self.starts[:-1]<=self.starts[1:]):
            raise ValueError('Train windows must be chronological for retrieval')
        self.history=ds.x_obs.shape[1];self.horizon=ds.y_norm.shape[1]
        self.shared=int(shared);self.pool_size=int(pool_size)
        n=len(ds); nodes=ds.x_obs.shape[-1]
        # Build train-only, source-valid signatures; no future Y or invisible x_full.
        self.signature=np.zeros((n,nodes,2),np.float32)
        self.visibility=np.zeros((n,nodes),bool)
        obs=ds.observed; artificial=ds.original_valid
        full=np.load(root/'train_x_full_norm.npy',mmap_mode='r')
        for a in range(0,n,512):
            b=min(n,a+512)
            m=np.asarray(obs[a:b],bool).copy()
            if artificial is not None:m|=np.asarray(artificial[a:b],bool)
            x=np.nan_to_num(np.asarray(full[a:b],np.float32))
            count=m.sum(1)
            self.visibility[a:b]=count>0
            self.signature[a:b,:,0]=(x*m).sum(1)/np.maximum(count,1)
            span=min(self.history,3)
            rc=m[:,-span:];r=rc.sum(1)
            self.signature[a:b,:,1]=(x[:,-span:]*rc).sum(1)/np.maximum(r,1)

    def select(self,x,mask,query_start,k=12):
        x=np.asarray(x,np.float32);m=np.asarray(mask,bool)&np.isfinite(x)
        x=np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        if x.shape!=m.shape or x.ndim!=2:raise ValueError('Query must be [L,N]')
        if x.shape[0]!=self.history or x.shape[1]!=self.signature.shape[1]:
            raise ValueError('Query shape mismatch')
        eligible=int(np.searchsorted(self.starts,int(query_start)-self.history-self.horizon,side='right'))
        if eligible<1:
            return np.zeros(k,np.int32),np.full(k,-1e4,np.float32)
        shared=min(self.shared,k,eligible)
        # Deterministic recent but well-separated anchors, irrespective of masks.
        legal=np.arange(eligible,dtype=np.int32)
        positions=np.linspace(eligible-1,max(0,eligible-1-16*(shared-1)),shared).round().astype(int)
        fixed=np.unique(legal[positions][::-1])
        if len(fixed)<shared:
            used=set(map(int,fixed));more=[p for p in range(eligible-1,-1,-1) if p not in used]
            fixed=np.concatenate([fixed,np.asarray(more[:shared-len(fixed)],np.int32)])
        fixed=fixed[:shared]
        recent=legal[max(0,eligible-self.pool_size//2):eligible]
        spaced=legal[np.linspace(0,eligible-1,min(eligible,self.pool_size),dtype=int)]
        pool=np.unique(np.concatenate([recent,spaced,fixed]))
        qcount=m.sum(0)
        latest=m[-min(3,self.history):]
        lastcount=latest.sum(0)
        q=np.stack([(x*m).sum(0)/np.maximum(qcount,1),
                    (x[-len(latest):]*latest).sum(0)/np.maximum(lastcount,1)],-1)
        qvalid=qcount>0
        both=self.visibility[pool]&qvalid[None]
        err=np.abs(self.signature[pool]-q[None]).mean(-1)
        overlap=both.sum(-1)
        dist=(err*both).sum(-1)/np.maximum(overlap,1)
        score=np.exp(-dist)-.05*(1-overlap/max(1,len(qvalid)))
        score=np.where(overlap>0,score,-.5)
        lookup={int(ix):float(s) for ix,s in zip(pool,score)}
        order=pool[np.argsort(-score,kind='stable')]
        chosen=list(map(int,fixed));used=set(chosen)
        for item in order:
            if len(chosen)>=k:break
            if int(item) not in used:chosen.append(int(item));used.add(int(item))
        valid_count=len(chosen)
        if valid_count<k:chosen += [chosen[0]]*(k-valid_count)
        scores=np.asarray([lookup.get(ix,-1e4) for ix in chosen],np.float32)
        scores[valid_count:]=-1e4
        return np.asarray(chosen,np.int32),scores
