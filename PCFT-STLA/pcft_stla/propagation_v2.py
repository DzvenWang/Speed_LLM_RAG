"""Train-only local-regime predictive influence, coherent temporal evidence and full-horizon support.
Edges are *statistical predictive associations*, never claimed road physics or causality.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def build_directional_graph(cache,neighbours,out,lags=(1,2,3,4,5,6),sampled_rows=1200):
    cache=Path(cache);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    nbr=np.asarray(neighbours.cpu() if torch.is_tensor(neighbours) else neighbours,dtype=np.int64)
    src=cache/'train_x_full_norm.npy'
    fingerprint=hashlib.sha256((str(src.resolve())+str(src.stat().st_size)+str(src.stat().st_mtime_ns)+
        hashlib.sha256(nbr.tobytes()).hexdigest()+str(tuple(lags))+str(sampled_rows)+'pcft-v2-local-innov').encode()).hexdigest()
    meta=out.with_suffix('.json')
    if out.is_file() and meta.is_file() and json.loads(meta.read_text()).get('fingerprint')==fingerprint:
        z=np.load(out);return torch.from_numpy(z['weights'].copy()),torch.from_numpy(z['confidence'].copy())
    data=np.load(src,mmap_mode='r')
    rows=np.linspace(0,len(data)-1,min(len(data),int(sampled_rows)),dtype=np.int64)
    x=np.nan_to_num(np.array(data[rows],dtype=np.float32,copy=True))
    b,h,n=x.shape;kn=nbr.shape[1]
    # Local regime on a target node, instead of network-averaged trend.
    trend=x[:,-3:].mean(1)-x[:,:3].mean(1)
    # Global thresholds computed on the Train sample only, separately for each node.
    lo=np.quantile(trend,1/3,axis=0);hi=np.quantile(trend,2/3,axis=0)
    label=(trend>lo[None]).astype(np.int8)+(trend>hi[None]).astype(np.int8)
    # Difference series suppresses spurious correlation from daily level changes.
    dx=np.diff(x.astype(np.float64),axis=1)
    w=np.zeros((3,n,kn,len(lags)),np.float32);conf=np.zeros_like(w)
    for ni in range(n):
        source=dx[:,:,nbr[ni]]  # [B,H-1,K]
        target=dx[:,:,ni]       # [B,H-1]
        for r in range(3):
            selected=label[:,ni]==r
            if int(selected.sum())<8:selected=np.ones(b,dtype=bool)
            for li,lag in enumerate(lags):
                if lag<1 or lag>=h-1:raise ValueError('Lag outside available train history differences')
                s=source[selected,:-lag,:].reshape(-1,kn)
                t=target[selected,lag:].reshape(-1)
                if len(t)<8:continue
                sc=s-s.mean(0,keepdims=True);tc=t-t.mean()
                cov=(sc*tc[:,None]).mean(0)
                sd=np.sqrt((sc*sc).mean(0)*(tc*tc).mean()+1e-8)
                corr=np.nan_to_num(cov/sd).clip(-1,1)
                # Shrink influence confidence for low effective sample counts.
                c=np.abs(corr)*np.sqrt(len(t)/(len(t)+250.))
                w[r,ni,:,li]=corr*c
                conf[r,ni,:,li]=c
    np.savez_compressed(out,weights=w,confidence=conf,neighbor=nbr,lags=np.asarray(lags,np.int16))
    meta.write_text(json.dumps({'fingerprint':fingerprint,'meaning':'local-regime differenced predictive association',
         'provenance':'train_x_full_norm only','lags':list(lags),'causal':False},indent=2))
    return torch.from_numpy(w),torch.from_numpy(conf)


class PropagationSupportV2(nn.Module):
    def __init__(self,num_nodes,neighbors,lags=(1,2,3,4,5,6)):
        super().__init__()
        self.register_buffer('neighbors',torch.as_tensor(neighbors,dtype=torch.long).clone())
        self.register_buffer('lags',torch.as_tensor(lags,dtype=torch.long))
        n,k=self.neighbors.shape;L=len(lags)
        self.register_buffer('weights',torch.zeros((3,n,k,L)))
        self.register_buffer('confidence',torch.zeros((3,n,k,L)))
        self.local_gate=nn.Sequential(nn.Linear(4,24),nn.GELU(),nn.Linear(24,3))
        nn.init.zeros_(self.local_gate[-1].weight);nn.init.zeros_(self.local_gate[-1].bias)

    def install(self,weights,confidence):
        if tuple(weights.shape)!=tuple(self.weights.shape):raise ValueError('Graph shape mismatch')
        self.weights.copy_(weights.to(self.weights));self.confidence.copy_(confidence.to(self.confidence))

    def active_graph(self,x=None,mask=None,batch_size=None):
        if x is None or getattr(self,'force_global',False):
            b=batch_size or 1
            return (self.weights.mean(0)[None].expand(b,-1,-1,-1),
                    self.confidence.mean(0)[None].expand(b,-1,-1,-1))
        m=mask.float(); h=x.shape[1];span=max(1,h//4)
        count=m.sum(1).clamp_min(1)
        level=(x*m).sum(1)/count
        lo=(x[:,:span]*m[:,:span]).sum(1)/m[:,:span].sum(1).clamp_min(1)
        hi=(x[:,-span:]*m[:,-span:]).sum(1)/m[:,-span:].sum(1).clamp_min(1)
        cover=m.mean(1)
        neighbor_coverage=cover[:,self.neighbors].mean(-1)
        features=torch.stack([hi-lo,level,cover,neighbor_coverage],-1)
        prob=torch.softmax(self.local_gate(features.float()),-1)   # [B,N,3]
        return (torch.einsum('bnr,rnkl->bnkl',prob,self.weights),
                torch.einsum('bnr,rnkl->bnkl',prob,self.confidence))

    def signature(self,x,mask):
        # 12-D stable memory-query index; no query future is used.
        b,h,n=x.shape;m=mask.float();observed=m.mean(1)
        level=(x*m).sum(1)/m.sum(1).clamp_min(1)
        change=(x[:,-1]-x[:,0])*(m[:,-1]*m[:,0])
        wg,conf=self.active_graph(x,mask)
        influence=(change[:,self.neighbors]*wg.sum(-1)).sum(-1)/conf.sum((-1,-2)).clamp_min(1)
        f=[]
        for v in [level,change,influence,observed]:
            f.extend([v.mean(1),v.std(1,unbiased=False),v.abs().mean(1)])
        return torch.stack(f,-1).float().clamp(-30,30)

    def support_field(self,mask,horizon,query_x=None,base_forecast=None,uniform_graph=False,lag_override=None):
        b,h,n=mask.shape;p=int(horizon)
        _,confidence=self.active_graph(query_x,mask,batch_size=b)
        confidence=confidence.abs()
        if uniform_graph:
            # Preserve the same neighbor set, remove only directional strength.
            confidence=torch.ones_like(confidence)
        neighbour=self.neighbors
        m=mask.float()
        observed=torch.zeros(b,p,n,device=mask.device,dtype=torch.float32)
        forecast=torch.zeros_like(observed)
        denom=confidence.sum((-1,-2)).clamp_min(1e-6)[:,None]
        for li,lag in enumerate(self.lags.tolist()):
            strength=confidence[:,:,:,li]
            for pi in range(p):
                t=h+pi-(int(lag_override) if lag_override is not None else lag)
                if 0<=t<h:
                    observed[:,pi]+=(m[:,t,neighbour]*strength).sum(-1)
                elif t>=h and base_forecast is not None and t<h+p:
                    # Forecast is a model *estimate*, never a true future observation.
                    # Horizon discount prevents self-confirming high confidence.
                    norm=torch.exp(-base_forecast[:,t-h,neighbour].abs()/10.)
                    forecast[:,pi]+=(norm*strength).sum(-1)*float(np.exp(-(pi+1)/12.))
        obs=(observed/denom).clamp(0,1)
        f=(forecast/denom).clamp(0,1)
        return {'observed':obs,'forecast':f,'uncertainty':(1-obs)*(1-f)}

    def support(self,mask,horizon,query_x=None,base_forecast=None):
        d=self.support_field(mask,horizon,query_x,base_forecast)
        return (d['observed']+0.5*d['forecast']).clamp(0,1)

    def transfer(self,history,future,horizon,query_x=None,query_mask=None,provenance=None,uniform_graph=False,lag_override=None):
        b,c,h,n=history.shape;p=int(horizon)
        origin=history[:,:,-1:,:]
        # historical future innovation on one coherent historical case
        both=torch.cat([history-origin,future-origin],2)
        weight,confidence=self.active_graph(query_x,query_mask,batch_size=b)
        output=torch.zeros(b,c,p,n,device=future.device,dtype=torch.float32)
        denom=torch.zeros_like(output)
        if provenance is not None and tuple(provenance.shape)!=(b,c,n):
            raise ValueError('Expected candidate provenance [B,C,N]')
        if uniform_graph:
            # Uniform over the *same* valid neighbors/lag support, preserving
            # graph topology and candidate provenance while removing edge scores.
            weight=torch.ones_like(weight)/max(1,weight.shape[-2])
            confidence=torch.ones_like(confidence)
        for li,lag in enumerate(self.lags.tolist()):
            for pi in range(p):
                t=h+pi-(int(lag_override) if lag_override is not None else lag)
                if not (0<=t<h+p):continue
                values=both[:,:,t,:]
                take=values[:,:,self.neighbors]
                coeff=weight[:,None,:,:,li].expand(b,c,-1,-1)
                trust=confidence[:,None,:,:,li].expand(b,c,-1,-1)
                if provenance is not None:
                    coherent=provenance[:,:,self.neighbors].eq(provenance[:,:,:,None])
                    coeff=coeff*coherent
                    trust=trust*coherent
                output[:,:,pi]+=(take*coeff).sum(-1)
                denom[:,:,pi]+=trust.sum(-1)
        return torch.where(denom>1e-8,output/denom.clamp_min(1e-8),torch.zeros_like(output))
