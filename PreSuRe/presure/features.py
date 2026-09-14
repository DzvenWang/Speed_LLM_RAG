from __future__ import annotations
import numpy as np

def _proj(n,d,seed):
    rng=np.random.default_rng(seed); R=rng.standard_normal((n,d),dtype=np.float32)/np.sqrt(max(n,1)); return R.astype(np.float32)

def build_features(x,obs,base,tod,dow,node_proj_dim=24,mask_proj_dim=12,seed=10,batch=512):
    """Deterministic sensing-aware descriptors; no future target is used."""
    x=np.asarray(x); obs=np.asarray(obs,bool); base=np.asarray(base)
    M,H,N=x.shape; P=base.shape[1]
    Rv=_proj(N,node_proj_dim,seed); Rm=_proj(N,mask_proj_dim,seed+1)
    D=H*3+node_proj_dim+P*2+node_proj_dim+4+H+mask_proj_dim+1
    out=np.empty((M,D),np.float32)
    for a in range(0,M,batch):
        z=slice(a,min(M,a+batch)); xx=np.asarray(x[z],np.float32); mm=np.asarray(obs[z],bool); bb=np.asarray(base[z],np.float32)
        den=np.maximum(mm.sum(2),1)
        tm=(xx*mm).sum(2)/den
        var=(((xx-tm[:,:,None])**2)*mm).sum(2)/den
        ts=np.sqrt(np.maximum(var,0))
        tf=mm.mean(2).astype(np.float32)
        nd=np.maximum(mm.sum(1),1); nm=(xx*mm).sum(1)/nd; npv=nm@Rv
        bm=bb.mean(2); bs=bb.std(2); bnode=bb.mean(1)@Rv
        lasttod=np.asarray(tod[z,-1],np.float32); theta=2*np.pi*lasttod/288.0
        lastdow=np.asarray(dow[z,-1],np.float32); phi=2*np.pi*lastdow/7.0
        tfeat=np.stack([np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)],1).astype(np.float32)
        nmask=mm.mean(1).astype(np.float32)@Rm; gmask=mm.mean((1,2),keepdims=False)[:,None].astype(np.float32)
        out[z]=np.concatenate([tm,ts,tf,npv,bm,bs,bnode,tfeat,tf,nmask,gmask],1)
    return out

def split_feature_dims(H,P,node_proj_dim,mask_proj_dim):
    # first part excludes explicit mask-only tail; tf appears once as state confidence and once in mask tail by design.
    state=H*3+node_proj_dim+P*2+node_proj_dim+4
    mask=H+mask_proj_dim+1
    return state,mask

def make_retrieval_keys(features,H,P,node_proj_dim,mask_proj_dim,mask_weight=0.35):
    s,m=split_feature_dims(H,P,node_proj_dim,mask_proj_dim)
    x=np.asarray(features,np.float32).copy(); x[:,s:s+m]*=float(mask_weight)
    mu=x.mean(0,keepdims=True); sd=x.std(0,keepdims=True)+1e-5; z=(x-mu)/sd
    n=np.linalg.norm(z,axis=1,keepdims=True)+1e-8
    return (z/n).astype(np.float32),mu.astype(np.float32),sd.astype(np.float32)

def apply_key_norm(features,mu,sd,H,P,node_proj_dim,mask_proj_dim,mask_weight=0.35):
    s,m=split_feature_dims(H,P,node_proj_dim,mask_proj_dim); x=np.asarray(features,np.float32).copy(); x[:,s:s+m]*=float(mask_weight)
    z=(x-mu)/(sd+1e-8); return (z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-8)).astype(np.float32)
