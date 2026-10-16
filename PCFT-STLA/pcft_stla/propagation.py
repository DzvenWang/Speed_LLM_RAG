"""Train-only directional predictive-lag association and horizon-aware support.

Directed correlations below are *statistical predictive influence*, not
physical topology or causal shockwave velocity. No test future enters here.
"""
from __future__ import annotations
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def build_directional_graph(train_cache,neighbours,out,lags=(1,2,3,4,5,6),sampled_rows=1200):
    cache=Path(train_cache);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    neighbor=np.asarray(neighbours.cpu() if torch.is_tensor(neighbours) else neighbours,dtype=np.int64)
    tfile=cache/'train_x_full_norm.npy'
    ident=hashlib.sha256((str(tfile.resolve())+str(tfile.stat().st_size)+str(tfile.stat().st_mtime_ns)+str(tuple(lags))+
                          hashlib.sha256(neighbor.tobytes()).hexdigest()+str(sampled_rows)+'pcft_v1').encode()).hexdigest()
    meta=out.with_suffix('.json')
    if out.exists() and meta.exists() and json.loads(meta.read_text()).get('fingerprint')==ident:
        z=np.load(out)
        return torch.from_numpy(z['weights'].copy()),torch.from_numpy(z['confidence'].copy())
    src=np.load(tfile,mmap_mode='r')
    ids=np.linspace(0,len(src)-1,min(len(src),int(sampled_rows)),dtype=np.int64)
    x=np.nan_to_num(np.array(src[ids],dtype=np.float32,copy=True))
    b,h,n=x.shape;k=neighbor.shape[1]
    # Three Train-only regimes: falling, stable, rising network-level state.
    # These are statistical evolution groups, not manually asserted physical
    # congestion labels. Each group builds a separate directional lag graph.
    trend=(x[:,-3:,:].mean((1,2))-x[:,:3,:].mean((1,2)))
    q1,q2=np.quantile(trend,[1/3,2/3])
    labels=np.digitize(trend,[q1,q2])
    weights=np.zeros((3,n,k,len(lags)),np.float32)
    for r in range(3):
        xr=x[labels==r]
        if len(xr)<4:xr=x
        for t,lag in enumerate(lags):
            if lag<1 or lag>=h:raise ValueError('Propagation lag must be >=1 and < history')
            source=xr[:,:h-lag,neighbor].reshape(-1,n,k).astype(np.float64)
            target=xr[:,lag:,:].reshape(-1,n).astype(np.float64)
            zs=source-source.mean(0)[None]
            zt=target-target.mean(0)[None]
            cov=(zs*zt[:,:,None]).mean(0)
            sd=np.sqrt((zs*zs).mean(0)*((zt*zt).mean(0)[:,None])+1e-8)
            weights[r,:,:,t]=np.clip(cov/sd,-1,1)
    confidence=np.abs(weights).astype(np.float32)
    # Retain signed correlation separately; absolute evidence is used for
    # support confidence, while the signed coefficient governs alignment.
    np.savez_compressed(out,weights=weights,confidence=confidence,neighbor=neighbor,
                        lags=np.asarray(lags,dtype=np.int16))
    meta.write_text(json.dumps({'fingerprint':ident,'source':'train_x_full_norm',
         'meaning':'lagged predictive associations, not physical causal road directions',
         'lags':list(map(int,lags)), 'regime_groups':'train-only falling/stable/rising network trends'},indent=2))
    return torch.from_numpy(weights),torch.from_numpy(confidence)


class PropagationSupport(nn.Module):
    """Sparse neighbor/time operation; never creates dense [N,N,P] tensors."""
    def __init__(self,num_nodes,neighbors,lags=(1,2,3,4,5,6)):
        super().__init__()
        self.P=12
        self.register_buffer('neighbors',torch.as_tensor(neighbors,dtype=torch.long).clone(),persistent=True)
        self.register_buffer('lags',torch.as_tensor(lags,dtype=torch.long),persistent=True)
        n,k=self.neighbors.shape;L=len(lags)
        self.register_buffer('weights',torch.zeros((3,n,k,L)),persistent=True)
        self.regime_gate=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,3))
        nn.init.zeros_(self.regime_gate[-1].weight)
        nn.init.zeros_(self.regime_gate[-1].bias)
        self.register_buffer('confidence',torch.zeros((3,n,k,L)),persistent=True)
        self.utility=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,1))
        self.expected=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,1))
        for layer in (self.utility[-1],self.expected[-1]):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def install(self,weights,confidence):
        if weights.ndim==3:
            weights=weights[None].expand(3,-1,-1,-1)
            confidence=confidence[None].expand(3,-1,-1,-1)
        if weights.shape!=self.weights.shape:raise ValueError('Propagation graph shape mismatch')
        self.weights.copy_(weights.to(self.weights));self.confidence.copy_(confidence.to(self.confidence))

    def active_graph(self, x=None, mask=None, batch_size=None):
        """Regime-conditioned signed predictive influence per Query, [B,N,K,L]."""
        if x is None:
            if batch_size is None:batch_size=1
            weights=self.weights.mean(0)[None].expand(batch_size,-1,-1,-1)
            conf=self.confidence.mean(0)[None].expand(batch_size,-1,-1,-1)
            return weights,conf
        m=mask.float()
        first=(x[:,:3]*m[:,:3]).sum((1,2))/m[:,:3].sum((1,2)).clamp_min(1.)
        last=(x[:,-3:]*m[:,-3:]).sum((1,2))/m[:,-3:].sum((1,2)).clamp_min(1.)
        coverage=m.mean((1,2))
        raw=torch.stack([last-first,last,coverage],-1).float()
        probs=torch.softmax(self.regime_gate(raw),dim=-1)
        return torch.einsum('br,rnkl->bnkl',probs,self.weights),torch.einsum('br,rnkl->bnkl',probs,self.confidence)

    def signature(self,x,mask):
        """[B,12] learned-free history descriptors, usable as memory index."""
        with torch.autocast(device_type=x.device.type,enabled=False):
            x=x.float();mask=mask.float();b,h,n=x.shape
            ix=self.neighbors
            observed=mask.mean(1)
            cov=observed[:,ix].mean(-1)
            l=x[:,-1];begin=x[:,0]
            trend=(l-begin)*(mask[:,-1]*mask[:,0])
            weights,confidence=self.active_graph(x,mask)
            influence=(trend[:,ix]*weights.sum(-1)).mean(-1)
            weighted=((l[:,ix]-l[:,:,None])*confidence.sum(-1)).mean(-1)
            f=[(trend*observed).mean(1),(influence*cov).mean(1),
               (weighted*cov).mean(1),observed.mean(1),cov.mean(1),
               (observed*cov).mean(1)]
            for v in (trend,influence,weighted):
                f.append(v.abs().mean(1));f.append(v.std(1,unbiased=False))
            return torch.stack(f,-1).float().clamp(-30,30)

    def support(self,mask,horizon,query_x=None):
        """Actual observed source support for each [B,P,N]. Future not read."""
        b,h,n=mask.shape;p=int(horizon)
        src=mask.float()[:,:,self.neighbors]  # B,H,N,K
        _,conf=self.active_graph(query_x,mask,batch_size=b)
        conf=conf.abs()
        numerator=torch.zeros(b,p,n,device=mask.device,dtype=torch.float32)
        denominator=conf.sum((-1,-2)).clamp_min(1e-6)[:,None]
        for li,lag in enumerate(self.lags.tolist()):
            for pi in range(p):
                t=h+pi-lag
                if 0<=t<h:
                    numerator[:,pi]+= (src[:,t]*conf[:,:,:,li]).sum(-1)
        return (numerator/denominator).clamp(0,1)

    def transfer(self,history,future,horizon,query_x=None,query_mask=None):
        """Signed lag-aligned future evolution, shape [B,C,P,N]."""
        b,c,h,n=history.shape;p=int(horizon)
        baseline=history[:,:,-1:,:]
        both=torch.cat([history-baseline,future-baseline],dim=2)
        neigh=self.neighbors
        combined=torch.zeros((b,c,p,n),device=future.device,dtype=torch.float32)
        weights,conf=self.active_graph(query_x,query_mask,batch_size=b)
        den=conf.sum((-1,-2)).clamp_min(1e-6)
        for li,lag in enumerate(self.lags.tolist()):
            for pi in range(p):
                t=h+pi-lag
                if 0<=t<both.shape[2]:
                    v=both[:,:,t,:][:,:,:,None].expand(b,c,n,neigh.shape[1])
                    # Gather along N using the candidate source-node index.
                    src=torch.gather(v,2,neigh[None,None].expand(b,c,-1,-1))
                    combined[:,:,pi]+=(src*weights[:,None,:,:,li]).sum(-1)
        return combined/den[:,None,None]

    def blend(self,legacy,prop,observability,score):
        b,c,p,n=legacy.shape
        supp=observability[:,None].expand(b,c,p,n)
        features=torch.stack([legacy,prop,supp,score[:,:,None].expand(b,c,p,n)],-1)
        # Bounded zero-initialized additive correction, so before training the
        # old Anchor/Seasonal Innovation path is reproduced exactly.
        beta=0.5*torch.tanh(self.utility(features).squeeze(-1))*supp
        corrected=legacy+beta*(prop-legacy)
        return corrected,beta,self.expected(features).squeeze(-1)
