"""Coarse Pearson shortlist over Train memory with temporal exclusion.

Ported from Mask_RAFT-STLA (retrieval.py) with two changes: memory features use
the Train original-availability mask (never-observed entries are not read as
zeros), and Train queries additionally use an embargo so that training-time
neighbours are not near-duplicates of the query window.
"""
from __future__ import annotations
import numpy as np
import torch


def coarse(values,mask,g):
    """values/mask [B,H,N] -> period-g averages [B,(H/g)*N] and their coverage."""
    B,H,N=values.shape
    if H%g:raise ValueError(f'history={H} not divisible by period={g}')
    v=values.float().view(B,H//g,g,N)
    if mask is None:
        return v.mean(2).reshape(B,-1),torch.ones((B,(H//g)*N),device=values.device)
    m=mask.float().view(B,H//g,g,N);den=m.sum(2)
    c=(v*m).sum(2)/den.clamp_min(1.0)
    return c.reshape(B,-1),(den/float(g)).reshape(B,-1)


def pearson(q,qm,x,min_valid=8):
    """Weighted Pearson over the query's observed coarse dimensions. q,qm [B,F]; x [C,F]."""
    qv=(qm>0).float()
    n=qv.sum(1,keepdim=True).clamp_min(1.0)
    sq=(q*qv).sum(1,keepdim=True);sq2=(q*q*qv).sum(1,keepdim=True)
    sx=qv@x.t();sx2=qv@(x*x).t();sqx=(q*qv)@x.t()
    cov=sqx-sq*sx/n;vq=(sq2-sq*sq/n).clamp_min(1e-8);vx=(sx2-sx*sx/n).clamp_min(1e-8)
    corr=torch.nan_to_num(cov/torch.sqrt(vq*vx),nan=-1.0,posinf=1.0,neginf=-1.0)
    return torch.where(qv.sum(1,keepdim=True)>=float(min_valid),corr,torch.full_like(corr,-1.0))


class MemoryIndex:
    def __init__(self,memory,periods,device,chunk=4096):
        self.memory=memory;self.periods=tuple(int(p) for p in periods);self.device=device
        if not self.periods:raise ValueError('at least one retrieval period required')
        feats={p:[] for p in self.periods}
        for s in range(0,len(memory),int(chunk)):
            rows=memory.rows(np.arange(s,min(len(memory),s+int(chunk))))
            x=torch.from_numpy(rows['x']);m=torch.from_numpy(rows['ov'])
            for p in self.periods:
                f,_=coarse(x,m,p);feats[p].append(f)
        self.features={p:torch.cat(v,0).to(device) for p,v in feats.items()}
        self.starts=torch.as_tensor(memory.starts,device=device,dtype=torch.long)

    def query(self,x,mask,starts,top_k,span,embargo=0,min_valid=8,chunk=8192):
        """Return idx [B,K] (-1 where fewer than K legal neighbours) and sim [B,K]."""
        x=x.to(self.device).float();mask=mask.to(self.device).float()
        qs=torch.as_tensor(starts,device=self.device,dtype=torch.long).view(-1)
        B=x.shape[0];K=int(top_k);gap=int(span)+int(embargo)
        best_s=torch.full((B,K),-float('inf'),device=self.device)
        best_i=torch.full((B,K),-1,dtype=torch.long,device=self.device)
        qf={p:coarse(x,mask,p) for p in self.periods}
        M=len(self.memory)
        for ms in range(0,M,int(chunk)):
            me=min(M,ms+int(chunk))
            score=None
            for p in self.periods:
                q,qm=qf[p]
                s=pearson(q,qm,self.features[p][ms:me],min_valid)
                score=s if score is None else score+s
            score=score/len(self.periods)
            illegal=(self.starts[ms:me][None]-qs[:,None]).abs()<gap
            score=score.masked_fill(illegal,-float('inf'))
            cand_s=torch.cat([best_s,score],1)
            cand_i=torch.cat([best_i,torch.arange(ms,me,device=self.device)[None].expand(B,-1)],1)
            best_s,pos=torch.topk(cand_s,k=K,dim=1)
            best_i=torch.gather(cand_i,1,pos)
        bad=~torch.isfinite(best_s)
        best_i=best_i.masked_fill(bad,-1)
        best_s=best_s.masked_fill(bad,0.0)
        return best_i,best_s


def legal_wrong_indices(query_starts,valid_k,train_starts,gap):
    """Deterministic, batch-independent Train cases that are NOT the retrieved ones
    and do not overlap the query (negative control 'cross_history')."""
    starts=np.asarray(train_starts,dtype=np.int64);q=np.asarray(query_starts,dtype=np.int64)
    valid_k=np.asarray(valid_k,dtype=bool)
    B,K=valid_k.shape;n=len(starts)
    out=np.full((B,K),-1,dtype=np.int64)
    for b in range(B):
        for k in range(K):
            if not valid_k[b,k]:continue
            j=(int(q[b])*1103515245+(k+1)*2654435761+0x9e3779b9)%n
            for _ in range(n):
                if abs(int(starts[j])-int(q[b]))>=gap:break
                j=(j+1)%n
            else:raise RuntimeError('No legal non-overlapping Train case for negative control')
            out[b,k]=j
    return out
