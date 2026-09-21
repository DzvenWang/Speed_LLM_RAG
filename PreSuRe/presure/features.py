from __future__ import annotations
import numpy as np

SENSE_DIM=6


def _proj(n,d,seed):
    rng=np.random.default_rng(seed); return (rng.standard_normal((n,d),dtype=np.float32)/np.sqrt(max(n,1))).astype(np.float32)


def row_normalize_adjacency(adj):
    a=np.asarray(adj,np.float32).copy()
    if a.ndim!=2 or a.shape[0]!=a.shape[1]: raise ValueError(f'Adjacency must be square, got {a.shape}')
    np.fill_diagonal(a,0.0); a=np.maximum(a,0.0); rs=a.sum(1,keepdims=True); iso=rs[:,0]<=1e-12
    if np.any(iso): a[iso,iso]=1.0; rs=a.sum(1,keepdims=True)
    return (a/np.maximum(rs,1e-8)).astype(np.float32)


def _masked_slope(x,obs):
    H=x.shape[1]; tt=np.linspace(-1.0,1.0,H,dtype=np.float32)[None,:,None]; mm=obs.astype(np.float32); den=(mm*tt*tt).sum(1).clip(min=1e-6)
    return ((x*mm*tt).sum(1)/den).astype(np.float32)


def _trailing_missing(obs):
    rev=(~obs[:,::-1,:]).astype(np.int16); return np.cumprod(rev,axis=1).sum(1).astype(np.float32)/max(obs.shape[1],1)


def _longest_missing(obs):
    B,H,N=obs.shape; cur=np.zeros((B,N),np.int16); best=np.zeros((B,N),np.int16)
    for h in range(H):
        miss=~obs[:,h,:]; cur=np.where(miss,cur+1,0); best=np.maximum(best,cur)
    return best.astype(np.float32)/max(H,1)


def _time_features(tod,dow):
    t=np.asarray(tod[:,-1],np.float32); d=np.asarray(dow[:,-1],np.float32); th=2*np.pi*t/288.0; ph=2*np.pi*d/7.0
    return np.stack([np.sin(th),np.cos(th),np.sin(ph),np.cos(ph)],1).astype(np.float32)


def build_neutral_coarse_descriptor(x,obs,tod,dow,node_proj_dim=32,mask_proj_dim=16,seed=10,batch=512):
    x=np.asarray(x); obs=np.asarray(obs,bool); M,H,N=x.shape; Rx=_proj(N,node_proj_dim,seed); Rm=_proj(N,mask_proj_dim,seed+1)
    D=H*node_proj_dim+H*mask_proj_dim+H*2+4; out=np.empty((M,D),np.float32)
    for a in range(0,M,int(batch)):
        z=slice(a,min(M,a+int(batch))); xx=np.nan_to_num(np.asarray(x[z],np.float32),nan=0.0); mf=np.asarray(obs[z],bool).astype(np.float32)
        out[z]=np.concatenate([np.einsum('bhn,nd->bhd',xx,Rx).reshape(len(xx),-1),np.einsum('bhn,nd->bhd',mf,Rm).reshape(len(xx),-1),xx.mean(2),xx.std(2),_time_features(tod[z],dow[z])],1)
    return np.nan_to_num(out,nan=0.0,posinf=12.0,neginf=-12.0)


def build_evidence_descriptor(x,obs,tod,dow,adj_norm,node_proj_dim=24,mask_proj_dim=16,seed=10,batch=512):
    x=np.asarray(x); obs=np.asarray(obs,bool); M,H,N=x.shape; A=np.asarray(adj_norm,np.float32)
    Rx=_proj(N,node_proj_dim,seed); Rd=_proj(N,node_proj_dim,seed+1); Rm=_proj(N,mask_proj_dim,seed+2); Rs=_proj(N,mask_proj_dim,seed+3)
    D=H*node_proj_dim+max(H-1,0)*node_proj_dim+H*mask_proj_dim+H+mask_proj_dim*6+4; out=np.empty((M,D),np.float32)
    for a in range(0,M,int(batch)):
        z=slice(a,min(M,a+int(batch))); xx=np.nan_to_num(np.asarray(x[z],np.float32),nan=0.0); mm=np.asarray(obs[z],bool); mf=mm.astype(np.float32); xobs=xx*mf
        dx=np.diff(xobs,axis=1) if H>1 else np.zeros((len(xx),0,N),np.float32)
        xp=np.einsum('bhn,nd->bhd',xobs,Rx).reshape(len(xx),-1); dp=np.einsum('bhn,nd->bhd',dx,Rd).reshape(len(xx),-1) if H>1 else np.zeros((len(xx),0),np.float32); mp=np.einsum('bhn,nd->bhd',mf,Rm).reshape(len(xx),-1)
        trail=_trailing_missing(mm); longest=_longest_missing(mm); current=mf[:,-1,:]; trans=(mm[:,1:]!=mm[:,:-1]).mean(1).astype(np.float32) if H>1 else np.zeros((len(xx),N),np.float32)
        nbr_obs=(mf.reshape(-1,N)@A.T).reshape(len(xx),H,N).mean(1); slope=_masked_slope(xx,mm); denom=(mf.reshape(-1,N)@A.T).reshape(len(xx),H,N); nbr_num=(xobs.reshape(-1,N)@A.T).reshape(len(xx),H,N); nbr=nbr_num/np.maximum(denom,1e-5); physical=xobs[:,-1]-nbr[:,-1]
        sp=np.concatenate([s@Rs for s in [trail,longest,current,trans,nbr_obs,physical+slope]],1)
        out[z]=np.concatenate([xp,dp,mp,mf.mean(2),sp,_time_features(tod[z],dow[z])],1)
    return np.nan_to_num(out,nan=0.0,posinf=12.0,neginf=-12.0)


def build_backbone_descriptor(base_pred,hidden,speed_mean,speed_std,node_proj_dim=24,hidden_node_proj_dim=8,seed=10,batch=512):
    base=np.asarray(base_pred); M,P,N=base.shape; Rb=_proj(N,node_proj_dim,seed); hdim=0
    if hidden is not None:
        hshape=np.asarray(hidden).shape
        if len(hshape)!=3 or hshape[0]!=M or hshape[1]!=N: raise ValueError(f'Unexpected hidden shape {hshape}')
        hdim=hshape[2]*hidden_node_proj_dim+hshape[2]*2; Rh=_proj(N,hidden_node_proj_dim,seed+1)
    out=np.empty((M,P*node_proj_dim+P*2+hdim),np.float32); std=max(float(speed_std),1e-6)
    for a in range(0,M,int(batch)):
        z=slice(a,min(M,a+int(batch))); bb=(np.asarray(base[z],np.float32)-float(speed_mean))/std; parts=[np.einsum('bpn,nd->bpd',bb,Rb).reshape(len(bb),-1),bb.mean(2),bb.std(2)]
        if hidden is not None:
            hh=np.asarray(hidden[z],np.float32); hp=np.einsum('bnc,nd->bcd',hh,Rh).reshape(len(hh),-1); parts.extend([hp,hh.mean(1),hh.std(1)])
        out[z]=np.concatenate(parts,1)
    return np.nan_to_num(out,nan=0.0,posinf=12.0,neginf=-12.0)


def fit_standardizer(x,clip=12.0):
    xx=np.asarray(x,np.float32); return xx.mean(0,keepdims=True).astype(np.float32),(xx.std(0,keepdims=True)+1e-5).astype(np.float32),float(clip)


def apply_standardizer(x,mu,sd,clip=12.0):
    z=(np.asarray(x,np.float32)-mu)/(sd+1e-8); return np.clip(np.nan_to_num(z,nan=0.0,posinf=clip,neginf=-clip),-clip,clip).astype(np.float32)


def build_node_dual_features_batch(x,obs,base_raw,hidden,tod,dow,adj_norm,speed_mean,speed_std):
    """Return state/evidence node descriptors [B,N,F]. Deployable inputs only."""
    xx=np.asarray(x,np.float32); mm=np.asarray(obs,bool); bb=np.asarray(base_raw,np.float32); B,H,N=xx.shape; P=bb.shape[1]; A=np.asarray(adj_norm,np.float32)
    mf=mm.astype(np.float32); xobs=xx*mf; std=max(float(speed_std),1e-6); bbn=(bb-float(speed_mean))/std
    bmean=bbn.mean(1); bstd=bbn.std(1); bslope=(bbn[:,-1]-bbn[:,0])/max(P-1,1)
    state_parts=[np.transpose(bbn,(0,2,1)),np.stack([bmean,bstd,bslope],-1)]
    if hidden is not None:
        hh=np.asarray(hidden,np.float32)
        if hh.ndim!=3 or hh.shape[:2]!=(B,N): raise ValueError(f'Unexpected node hidden {hh.shape}, expected {(B,N,"C")}')
        state_parts.append(hh)
    else:
        state_parts.append(np.zeros((B,N,1),np.float32))
    state=np.concatenate(state_parts,-1).astype(np.float32)

    num=(xobs.reshape(-1,N)@A.T).reshape(B,H,N); nobs=(mf.reshape(-1,N)@A.T).reshape(B,H,N); nbr=num/np.maximum(nobs,1e-5)
    dx=np.diff(xobs,axis=1,prepend=xobs[:,:1]); den=mm.sum(1).clip(min=1); nmean=xobs.sum(1)/den; nvar=(((xx-nmean[:,None])**2)*mf).sum(1)/den; nstd=np.sqrt(np.maximum(nvar,0)); slope=_masked_slope(xx,mm)
    obsfrac=mf.mean(1); trail=_trailing_missing(mm); longest=_longest_missing(mm); nbr_obs=nobs.mean(1); trans=(mm[:,1:]!=mm[:,:-1]).mean(1).astype(np.float32) if H>1 else np.zeros((B,N),np.float32); current=mf[:,-1]
    tfeat=np.repeat(_time_features(tod,dow)[:,None,:],N,axis=1)
    evidence=np.concatenate([
        np.transpose(xobs,(0,2,1)),np.transpose(mf,(0,2,1)),np.transpose(dx,(0,2,1)),np.transpose(nbr,(0,2,1)),np.transpose(nobs,(0,2,1)),
        np.stack([nmean,nstd,slope,nbr[:,-1],obsfrac,trail,longest,nbr_obs,trans,current],-1),tfeat
    ],-1).astype(np.float32)
    return np.nan_to_num(state,nan=0.0,posinf=8.0,neginf=-8.0),np.nan_to_num(evidence,nan=0.0,posinf=8.0,neginf=-8.0)
