from __future__ import annotations
import numpy as np
from .utils import log

def search_ip(memory,query,k):
    memory=np.ascontiguousarray(memory,np.float32); query=np.ascontiguousarray(query,np.float32)
    try:
        import faiss
        idx=faiss.IndexFlatIP(memory.shape[1]); idx.add(memory); scores,inds=idx.search(query,int(k)); return inds.astype(np.int32),scores.astype(np.float32),'faiss_cpu'
    except Exception as e:
        log(f'FAISS unavailable ({e!r}); using chunked numpy retrieval')
        all_i=[]; all_s=[]
        for a in range(0,len(query),128):
            s=query[a:a+128]@memory.T; ii=np.argpartition(-s,kth=min(k-1,s.shape[1]-1),axis=1)[:,:k]
            ss=np.take_along_axis(s,ii,1); o=np.argsort(-ss,axis=1); all_i.append(np.take_along_axis(ii,o,1)); all_s.append(np.take_along_axis(ss,o,1))
        return np.concatenate(all_i).astype(np.int32),np.concatenate(all_s).astype(np.float32),'numpy'

def filter_candidates(raw_i,raw_s,qmeta,memmeta,k,interval=24,train_query=False):
    out_i=np.full((len(qmeta),k),-1,np.int32); out_s=np.full((len(qmeta),k),-np.inf,np.float32)
    mg=memmeta.view_group_id.astype(str).to_numpy(); mw=memmeta.window_start.astype(int).to_numpy()
    qg=qmeta.view_group_id.astype(str).to_numpy(); qw=qmeta.window_start.astype(int).to_numpy()
    for q in range(len(qmeta)):
        n=0
        for j,s in zip(raw_i[q],raw_s[q]):
            if j<0: continue
            # Prevent self/same underlying episode and history+future interval overlap.
            if mg[j]==qg[q]: continue
            if abs(int(mw[j])-int(qw[q])) < int(interval): continue
            out_i[q,n]=j; out_s[q,n]=s; n+=1
            if n==k: break
        if n<k: raise RuntimeError(f'Only {n}/{k} leakage-safe candidates for query {q}; increase search_overfetch')
    return out_i,out_s
