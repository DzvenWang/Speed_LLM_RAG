from __future__ import annotations

from dataclasses import dataclass
import math,time
import torch

@dataclass(frozen=True)
class TMCPInfo:
    iterations:int; converged_fraction:float; max_relative_change:float; elapsed_seconds:float


def _unfold(x,mode):
    y=torch.movedim(x,mode+1,1); return y.reshape(y.shape[0],y.shape[1],-1)

def _fold(mat,shape,mode):
    others=[shape[i] for i in range(3) if i!=mode]
    y=mat.reshape(mat.shape[0],shape[mode],*others); return torch.movedim(y,1,mode+1)

def _shrink(s,lam,gamma):
    if gamma<=1: return torch.where(s>lam,s,torch.zeros_like(s))
    out=gamma/(gamma-1.0)*torch.relu(s-lam); return torch.where(s>gamma*lam,s,out)

def _transform(s,tau,theta,gamma):
    ss=_shrink(s,float(tau),float(gamma)); rank=s.shape[-1]
    th=theta.reshape(-1).to(s); r=torch.ceil(th*rank).long().clamp_(0,rank)
    keep=torch.arange(rank,device=s.device)[None,:] < r[:,None]
    return torch.where(keep,s,ss)

def _svt_gram(mat,tau,theta,gamma):
    _,m,n=mat.shape; eps=torch.finfo(mat.dtype).eps
    if m<=n:
        gram=mat@mat.transpose(-1,-2); vals,vec=torch.linalg.eigh(gram)
        vals=vals.flip(-1).clamp_min_(0); vec=vec.flip(-1); s=torch.sqrt(vals); ss=_transform(s,tau,theta,gamma)
        ratio=torch.where(s>eps,ss/s,torch.zeros_like(s)); return (vec*ratio.unsqueeze(-2))@(vec.transpose(-1,-2)@mat)
    gram=mat.transpose(-1,-2)@mat; vals,vec=torch.linalg.eigh(gram)
    vals=vals.flip(-1).clamp_min_(0); vec=vec.flip(-1); s=torch.sqrt(vals); ss=_transform(s,tau,theta,gamma)
    ratio=torch.where(s>eps,ss/s,torch.zeros_like(s)); return ((mat@vec)*ratio.unsqueeze(-2))@vec.transpose(-1,-2)

def _svt_svd(mat,tau,theta,gamma):
    u,s,vh=torch.linalg.svd(mat,full_matrices=False); ss=_transform(s,tau,theta,gamma); return (u*ss.unsqueeze(-2))@vh

@torch.no_grad()
def recover_batch(sparse_tensor,observed_mask,*,rho=1e-5,theta0=.2,beta=2.5,gamma=20.,factor=1.05,tol=1e-4,max_iter=200,clamp_min=0.,clamp_max=1e4,decomposition='gram_eigh'):
    if sparse_tensor.ndim!=4 or observed_mask.shape!=sparse_tensor.shape: raise ValueError('TMCP expects matching [B,D0,D1,D2]')
    t0=time.perf_counter(); x=sparse_tensor.float(); obs=observed_mask.bool(); B=x.shape[0]; shape=tuple(map(int,x.shape[1:])); dim=3
    of=obs.float(); den=of.flatten(1).sum(1).clamp_min(1.); means=(x*of).flatten(1).sum(1)/den
    M=torch.where(obs,x,means[:,None,None,None]).clone(); Z=torch.zeros((dim,*x.shape),device=x.device); T=torch.zeros_like(Z)
    theta=float(theta0)*torch.exp(-float(beta)*(~obs).float().flatten(1).mean(1)); rel=torch.full((B,),float('inf'),device=x.device); active=torch.ones(B,dtype=torch.bool,device=x.device); rho_t=float(rho)
    for it in range(int(max_iter)):
        ids=torch.nonzero(active,as_tuple=False).flatten()
        if ids.numel()==0: break
        prev=M[ids].clone(); rho_t=float(min(max(rho_t*float(factor),1e-10),1e5))
        for k in range(dim):
            mat=_unfold(M[ids]-T[k,ids]/rho_t,k)
            zk=_svt_gram(mat,1.0/3.0/rho_t,theta[ids],gamma) if decomposition=='gram_eigh' else _svt_svd(mat,1.0/3.0/rho_t,theta[ids],gamma)
            Z[k,ids]=_fold(zk,shape,k)
        prop=torch.mean(Z[:,ids]+T[:,ids]/rho_t,dim=0); upd=torch.where(obs[ids],x[ids],prop).clamp(float(clamp_min),float(clamp_max)); upd=torch.where(obs[ids],x[ids],upd)
        M[ids]=upd; T[:,ids]+=rho_t*(Z[:,ids]-M[ids].unsqueeze(0))
        num=torch.linalg.vector_norm((M[ids]-prev).flatten(1),dim=1); den2=torch.linalg.vector_norm(prev.flatten(1),dim=1).clamp_min(1e-12); r=num/den2; rel[ids]=r; active[ids[r<float(tol)]]=False
    finite=rel[torch.isfinite(rel)]
    info=TMCPInfo(it+1 if 'it' in locals() else 0,float((~active).float().mean()),float(finite.max()) if finite.numel() else float('inf'),time.perf_counter()-t0)
    return M,info
