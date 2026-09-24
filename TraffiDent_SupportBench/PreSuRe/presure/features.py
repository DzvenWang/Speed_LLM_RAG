from __future__ import annotations
import numpy as np

NODE_SENSE_DIM = 6


def _proj(n,d,seed):
    rng=np.random.default_rng(seed); R=rng.standard_normal((n,d),dtype=np.float32)/np.sqrt(max(n,1)); return R.astype(np.float32)


def build_features(x,obs,base,tod,dow,node_proj_dim=24,mask_proj_dim=12,seed=10,batch=512):
    """Global deterministic descriptor used only for high-recall coarse retrieval.

    No future target, direct-mask, support-mask, or outage-family label is used.
    """
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
        ts=np.sqrt(np.maximum(var,0)); tf=mm.mean(2).astype(np.float32)
        nd=np.maximum(mm.sum(1),1); nm=(xx*mm).sum(1)/nd; npv=nm@Rv
        bm=bb.mean(2); bs=bb.std(2); bnode=bb.mean(1)@Rv
        lasttod=np.asarray(tod[z,-1],np.float32); theta=2*np.pi*lasttod/288.0
        lastdow=np.asarray(dow[z,-1],np.float32); phi=2*np.pi*lastdow/7.0
        tfeat=np.stack([np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)],1).astype(np.float32)
        nmask=mm.mean(1).astype(np.float32)@Rm; gmask=mm.mean((1,2),keepdims=False)[:,None].astype(np.float32)
        out[z]=np.concatenate([tm,ts,tf,npv,bm,bs,bnode,tfeat,tf,nmask,gmask],1)
    return out


def split_feature_dims(H,P,node_proj_dim,mask_proj_dim):
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


def row_normalize_adjacency(adj):
    """Row-normalized neighbor matrix with self loops removed."""
    a=np.asarray(adj,np.float32).copy()
    if a.ndim!=2 or a.shape[0]!=a.shape[1]: raise ValueError(f'Adjacency must be square, got {a.shape}')
    np.fill_diagonal(a,0.0); a=np.maximum(a,0.0)
    rs=a.sum(1,keepdims=True); isolated=(rs[:,0]<=1e-12)
    if np.any(isolated):
        # Deterministic fallback: isolated nodes use self only for numerical stability.
        a[isolated,isolated]=1.0; rs=a.sum(1,keepdims=True)
    return (a/np.maximum(rs,1e-8)).astype(np.float32)


def _masked_slope(x,obs):
    """Least-squares slope over history for [B,H,N]."""
    H=x.shape[1]; tt=np.linspace(-1.0,1.0,H,dtype=np.float32)[None,:,None]
    mm=obs.astype(np.float32); den=(mm*tt*tt).sum(1).clip(min=1e-6)
    return ((x*mm*tt).sum(1)/den).astype(np.float32)


def _trailing_missing(obs):
    # obs [B,H,N], return normalized trailing missing length [B,N]
    rev=(~obs[:,::-1,:]).astype(np.int16)
    # cumulative product remains 1 only until the first observed position.
    return np.cumprod(rev,axis=1).sum(1).astype(np.float32)/max(obs.shape[1],1)


def _longest_missing(obs):
    B,H,N=obs.shape; cur=np.zeros((B,N),np.int16); best=np.zeros((B,N),np.int16)
    for h in range(H):
        miss=~obs[:,h,:]; cur=np.where(miss,cur+1,0); best=np.maximum(best,cur)
    return best.astype(np.float32)/max(H,1)


def build_node_features_batch(x,obs,base_raw,tod,dow,adj_norm,speed_mean,speed_std):
    """Sensing-conditioned node-local descriptors.

    Inputs are deployable quantities only: incomplete history, history observation mask,
    graph adjacency, time covariates, and frozen PatchSTG forecast. Evaluation-only
    direct/support masks are intentionally excluded.

    Returns [B,N,F]. The final NODE_SENSE_DIM columns are the explicit sensing state:
      obs_fraction, trailing_missing, longest_missing, neighbor_obs_fraction,
      mask_transition_rate, current_observed.
    """
    xx=np.asarray(x,np.float32); mm=np.asarray(obs,bool); bb=np.asarray(base_raw,np.float32)
    B,H,N=xx.shape; P=bb.shape[1]
    A=np.asarray(adj_norm,np.float32)
    if A.shape!=(N,N): raise ValueError(f'Adjacency {A.shape} incompatible with N={N}')
    # Missing values are never treated as real values: zero them before every aggregation.
    xobs=xx*mm.astype(np.float32)
    # Dense BLAS is substantially faster here than a generic einsum for N≈388.
    num=(xobs.reshape(-1,N) @ A.T).reshape(B,H,N)
    nobs=(mm.astype(np.float32).reshape(-1,N) @ A.T).reshape(B,H,N)
    nbr=num/np.maximum(nobs,1e-5)
    # PatchSTG prediction is raw speed; normalize with the Train cache statistics.
    bbn=(bb-float(speed_mean))/max(float(speed_std),1e-6)
    den=mm.sum(1).clip(min=1)
    nmean=xobs.sum(1)/den
    nvar=(((xx-nmean[:,None,:])**2)*mm).sum(1)/den
    nstd=np.sqrt(np.maximum(nvar,0)).astype(np.float32)
    slope=_masked_slope(xx,mm)
    nbr_obs=nobs.mean(1).astype(np.float32)
    nbr_last=nbr[:,-1,:].astype(np.float32)
    bmean=bbn.mean(1); bstd=bbn.std(1); bslope=(bbn[:,-1,:]-bbn[:,0,:])/max(P-1,1)
    lasttod=np.asarray(tod[:,-1],np.float32); th=2*np.pi*lasttod/288.0
    lastdow=np.asarray(dow[:,-1],np.float32); ph=2*np.pi*lastdow/7.0
    tfeat=np.stack([np.sin(th),np.cos(th),np.sin(ph),np.cos(ph)],1).astype(np.float32)
    tfeat=np.repeat(tfeat[:,None,:],N,axis=1)
    obsfrac=mm.mean(1).astype(np.float32)
    trail=_trailing_missing(mm); longest=_longest_missing(mm)
    trans=(mm[:,1:,:]!=mm[:,:-1,:]).mean(1).astype(np.float32) if H>1 else np.zeros((B,N),np.float32)
    current=mm[:,-1,:].astype(np.float32)
    sense=np.stack([obsfrac,trail,longest,nbr_obs,trans,current],-1).astype(np.float32)
    # [B,N,*]
    parts=[
        np.transpose(xobs,(0,2,1)),
        np.transpose(mm.astype(np.float32),(0,2,1)),
        np.transpose(nbr,(0,2,1)),
        np.transpose(nobs,(0,2,1)),
        np.transpose(bbn,(0,2,1)),
        np.stack([nmean,nstd,slope,nbr_obs,nbr_last,bmean,bstd,bslope],-1).astype(np.float32),
        tfeat,
        sense,
    ]
    out=np.concatenate(parts,-1).astype(np.float32)
    return np.nan_to_num(out,nan=0.0,posinf=8.0,neginf=-8.0)


def node_feature_dim(H,P):
    return 4*int(H)+int(P)+8+4+NODE_SENSE_DIM


def node_sensing_from_features(f):
    x=np.asarray(f)
    return x[...,-NODE_SENSE_DIM:].astype(np.float32)
