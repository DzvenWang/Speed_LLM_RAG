"""Outcome statistics of failure-replayed Train neighbours (pure tensor code).

For a query with the current mask M_q, every retrieved Train neighbour c was
re-forecast by the SAME frozen STLA after the query's failure pattern was
replayed on the neighbour's own history (``p0`` = P0_c(M_q)).  Together with
the neighbour's full-evidence forecast ``p_full`` = P_full_c and its label
``y`` this gives the exact decomposition, per neighbour,

    y - p0  =  (p_full - p0)  +  (y - p_full)
               restoration shift  normal residual

All features below are label-free for the QUERY: they only use Train labels of
the neighbours.  Shapes: sim/valid_k [B,K]; p0/p_full/y/fv [B,K,P,N];
r_hat/d_hat [B,P,N] (the parametric Normal residual and CF shift).
"""
from __future__ import annotations
import torch

RET_FEATURE_DIM=13
FEATURE_NAMES=('r_ret','d_ret','g_ret','sd_r','sd_d','harm_d','harm_r','agree_d',
               'abs_d_minus_cf','abs_r_minus_normal','log1p_neff','top_sim','coverage')


def neighbour_weights(sim,valid_k,tau):
    """Softmax over legal neighbours.  Rows without any legal neighbour -> 0."""
    sim=sim.float();valid_k=valid_k.bool()
    tau=torch.as_tensor(tau,dtype=sim.dtype,device=sim.device).clamp_min(1e-3)
    logits=torch.where(valid_k,sim/tau,torch.full_like(sim,-1e9))
    w=torch.softmax(logits,dim=1)*valid_k.to(sim.dtype)
    return w/w.sum(1,keepdim=True).clamp_min(1e-12)


def _element_weights(w,fv):
    """Window weights restricted to neighbours whose future element is valid."""
    W=w[:,:,None,None]*fv.to(w.dtype)
    mass=W.sum(1,keepdim=True)
    return W/mass.clamp_min(1e-12),mass.squeeze(1)


def _loo_harm(W,base,shift,y,fv,eps=1e-3):
    """Leave-one-out harm rate of transferring the neighbours' mean shift.

    For neighbour c the shift estimated from the OTHER neighbours is applied to
    c's own base forecast; harm_c = the correction increases c's absolute error.
    Returns the W-weighted harm probability [B,P,N] and the mass that had a
    defined leave-one-out estimate.  Elements with no estimate get 0.5.
    """
    total=(W*shift).sum(1,keepdim=True)
    rest=(1.0-W)
    ok=(rest>eps)&fv.bool()
    loo=torch.where(ok,(total-W*shift)/rest.clamp_min(eps),torch.zeros_like(shift))
    harm=((base+loo-y).abs()>(base-y).abs()).to(W.dtype)
    weight=W*ok.to(W.dtype)
    mass=weight.sum(1)
    rate=(weight*harm).sum(1)/mass.clamp_min(1e-12)
    return torch.where(mass>1e-6,rate,torch.full_like(rate,0.5)),mass


def evidence_stats(evidence,tau,r_hat,d_hat):
    """Return a dict with r_ret,d_ret,g_ret ([B,P,N], differentiable in tau),
    features [B,P,N,RET_FEATURE_DIM] (detached) and the window weights."""
    sim=evidence['sim'];valid_k=evidence['valid_k']
    p0=evidence['p0'].float();pf=evidence['p_full'].float()
    y=evidence['y'].float();fv=evidence['fv'].bool()&valid_k[:,:,None,None].bool()
    if p0.shape!=pf.shape or p0.shape!=y.shape or p0.shape!=fv.shape:
        raise ValueError('evidence tensors must share [B,K,P,N]')
    B,K,P,N=p0.shape
    if sim.shape!=(B,K) or valid_k.shape!=(B,K):raise ValueError('sim/valid_k must be [B,K]')
    y=torch.where(fv,y,torch.zeros_like(y))
    res=torch.where(fv,y-pf,torch.zeros_like(y))
    cf=torch.where(fv,pf-p0,torch.zeros_like(y))
    gain=torch.where(fv,(p0-y).abs()-(pf-y).abs(),torch.zeros_like(y))
    w=neighbour_weights(sim,valid_k,tau)
    W,coverage=_element_weights(w,fv)
    r_ret=(W*res).sum(1);d_ret=(W*cf).sum(1);g_ret=(W*gain).sum(1)
    with torch.no_grad():
        Wd=W.detach()
        sd_r=((Wd*(res-r_ret.detach()[:,None])**2).sum(1)).clamp_min(0).sqrt()
        sd_d=((Wd*(cf-d_ret.detach()[:,None])**2).sum(1)).clamp_min(0).sqrt()
        harm_d,_=_loo_harm(Wd,p0,cf,y,fv)
        harm_r,_=_loo_harm(Wd,pf,res,y,fv)
        rh=r_hat.detach().float();dh=d_hat.detach().float()
        agree=(torch.sign(d_ret.detach())*torch.sign(dh)+1.0)*0.5
        wd=w.detach()
        neff=1.0/(wd.square().sum(1)).clamp_min(1e-12)
        neff=torch.where(valid_k.any(1),neff,torch.zeros_like(neff))
        top=torch.where(valid_k.bool(),sim.float(),torch.full_like(sim.float(),-1.0)).max(1).values
        top=torch.where(valid_k.any(1),top,torch.full_like(top,-1.0))
        grid=lambda v:v[:,None,None].expand(B,P,N)
        features=torch.stack([r_ret.detach(),d_ret.detach(),g_ret.detach(),sd_r,sd_d,
                              harm_d,harm_r,agree,(d_ret.detach()-dh).abs(),
                              (r_ret.detach()-rh).abs(),torch.log1p(grid(neff)),grid(top),
                              coverage.detach().clamp(0,1)],dim=-1)
        features=torch.nan_to_num(features,nan=0.0,posinf=0.0,neginf=0.0)
    return {'r_ret':r_ret,'d_ret':d_ret,'g_ret':g_ret,'features':features,'weights':w}
