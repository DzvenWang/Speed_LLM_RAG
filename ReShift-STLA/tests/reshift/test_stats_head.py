"""Neighbour outcome statistics, arm selection and the ReShift head."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
from helpers import P,N
from reshift.retrieval.stats import RET_FEATURE_DIM,evidence_stats,neighbour_weights
from reshift.retrieval.head import RetrievalHead,select_arm,validate_arms
from reshift.counterfactual.dose import select_dose

torch.set_num_threads(1)


def evidence(B=3,K=4,seed=0,pad=True):
    g=torch.Generator().manual_seed(seed)
    ev={'sim':torch.rand(B,K,generator=g),'valid_k':torch.ones(B,K,dtype=torch.bool),
        'p0':torch.randn(B,K,P,N,generator=g),'p_full':torch.randn(B,K,P,N,generator=g),
        'y':torch.randn(B,K,P,N,generator=g),'fv':torch.rand(B,K,P,N,generator=g)>.2}
    if pad:ev['valid_k'][0,-1]=False
    return ev


def reference(ev,tau):
    sim=ev['sim'].numpy();vk=ev['valid_k'].numpy();p0=ev['p0'].numpy();pf=ev['p_full'].numpy()
    y=ev['y'].numpy();fv=ev['fv'].numpy()&vk[:,:,None,None]
    B,K=sim.shape;out={k:np.zeros((B,P,N)) for k in ('r','d','g','harm_d')}
    for b in range(B):
        logits=np.where(vk[b],sim[b]/tau,-np.inf);w=np.exp(logits-logits[vk[b]].max());w[~vk[b]]=0;w/=w.sum()
        for p in range(P):
            for n in range(N):
                m=fv[b,:,p,n];ww=w*m
                if ww.sum()<=0:out['harm_d'][b,p,n]=0.5;continue
                W=ww/ww.sum()
                res=y[b,:,p,n]-pf[b,:,p,n];cf=pf[b,:,p,n]-p0[b,:,p,n]
                gain=np.abs(p0[b,:,p,n]-y[b,:,p,n])-np.abs(pf[b,:,p,n]-y[b,:,p,n])
                out['r'][b,p,n]=(W*res).sum();out['d'][b,p,n]=(W*cf).sum();out['g'][b,p,n]=(W*gain).sum()
                num=0.;den=0.
                for c in range(K):
                    if not m[c] or 1-W[c]<=1e-3:continue
                    loo=((W*cf).sum()-W[c]*cf[c])/(1-W[c])
                    h=abs(p0[b,c,p,n]+loo-y[b,c,p,n])>abs(p0[b,c,p,n]-y[b,c,p,n])
                    num+=W[c]*h;den+=W[c]
                out['harm_d'][b,p,n]=num/den if den>1e-6 else 0.5
    return out


def test_evidence_stats_match_bruteforce():
    ev=evidence();tau=0.3
    z=torch.zeros(3,P,N)
    st=evidence_stats(ev,torch.tensor(tau),z,z)
    ref=reference(ev,tau)
    assert st['features'].shape==(3,P,N,RET_FEATURE_DIM)
    np.testing.assert_allclose(st['r_ret'].numpy(),ref['r'],atol=1e-5)
    np.testing.assert_allclose(st['d_ret'].numpy(),ref['d'],atol=1e-5)
    np.testing.assert_allclose(st['g_ret'].numpy(),ref['g'],atol=1e-5)
    np.testing.assert_allclose(st['features'][...,5].numpy(),ref['harm_d'],atol=1e-5)


def test_rows_without_neighbours_are_finite_zero():
    ev=evidence();ev['valid_k'][1]=False
    z=torch.zeros(3,P,N)
    st=evidence_stats(ev,torch.tensor(0.1),z,z)
    assert torch.isfinite(st['features']).all()
    assert torch.all(st['r_ret'][1]==0) and torch.all(st['d_ret'][1]==0)
    assert torch.allclose(st['features'][1,...,5],torch.full((P,N),0.5))
    w=neighbour_weights(ev['sim'],ev['valid_k'],0.1)
    assert torch.all(w[1]==0) and torch.allclose(w[0].sum(),torch.tensor(1.))


def test_tau_receives_gradient():
    ev=evidence();tau=torch.tensor(0.2,requires_grad=True)
    z=torch.zeros(3,P,N)
    st=evidence_stats(ev,tau,z,z)
    (st['r_ret'].sum()+st['d_ret'].sum()).backward()
    assert tau.grad is not None and torch.isfinite(tau.grad) and tau.grad.abs()>0


def test_select_arm_hard_soft_and_margin():
    g=torch.tensor([[[[0.1]],[[0.3]],[[-0.2]]]]).expand(1,3,2,2).clone()
    c=torch.tensor([[[[1.]],[[2.]],[[3.]]]]).expand(1,3,2,2).clone()
    applied,arm=select_arm(g,c)
    assert torch.all(applied==2.) and torch.all(arm==1)
    applied,arm=select_arm(g,c,margin=0.5)
    assert torch.all(applied==0.) and torch.all(arm==-1)
    applied,_=select_arm(g-10,c,temperature=0.01)
    assert torch.allclose(applied,torch.zeros_like(applied),atol=1e-6)
    with pytest.raises(ValueError):select_arm(g,c[:,:2])


def test_select_arm_reduces_to_select_dose():
    torch.manual_seed(0)
    levels=torch.tensor([0.25,0.5,1.0]);gains=torch.randn(2,3,P,N)
    psi=torch.rand(2,3,P,N);delta=torch.randn(2,P,N)
    psi_sel,_=select_dose(gains,psi,levels,margin=0.0)
    applied,_=select_arm(gains,psi*delta[:,None],margin=0.0)
    assert torch.allclose(applied,psi_sel*delta,atol=1e-6)


def test_head_arms_and_zero_init():
    torch.manual_seed(1)
    h=RetrievalHead(8,['cf','ret_res','ret_shift','ret_both'],dose_levels=(0.5,1.0),consistency_dim=4)
    assert h.arm_kinds==('cf','cf','ret_res','ret_shift','ret_both') and h.needs_evidence and h.has_fusion
    tok=torch.randn(2,P,N,8);d=torch.randn(2,P,N);r=torch.randn(2,P,N);psi=torch.rand(2,2,P,N)
    st=evidence_stats(evidence(B=2),h.tau(),r,d)
    corr,gates=h.corrections(tok,d,r,psi,st)
    assert corr.shape==(2,5,P,N) and gates.shape==(2,P,N,2)
    assert torch.allclose(corr[:,0],psi[:,0]*d) and torch.allclose(corr[:,1],psi[:,1]*d)
    assert torch.allclose(corr[:,4],corr[:,2]+corr[:,3],atol=1e-6)
    gains=h.gains(tok,corr,torch.randn(2,P,N),torch.randn(2,P,N,4),st['features'])
    assert gains.shape==(2,5,P,N) and torch.all(gains==0)


def test_head_variants():
    with pytest.raises(ValueError):validate_arms(['cf'],cf_enabled=False)
    with pytest.raises(ValueError):validate_arms(['cf','cf'])
    with pytest.raises(ValueError):validate_arms(['nope'])
    h=RetrievalHead(8,['ret_shift'],cf_enabled=False,use_features=False)
    tok=torch.randn(1,P,N,8);r=torch.zeros(1,P,N);d=torch.randn(1,P,N)
    st=evidence_stats(evidence(B=1),h.tau(),r,d)
    corr,gates=h.corrections(tok,d,r,None,st)
    assert torch.allclose(corr[:,0],gates[...,1]*st['d_ret'],atol=1e-6)   # no parametric CF used
    p=RetrievalHead(8,['cf','prior'],use_features=False)
    assert not p.needs_evidence and p.has_fusion and p.gate is None
    corr,_=p.corrections(tok,d,r,torch.ones(1,1,P,N),None)
    assert torch.all(corr[:,1]==0)                                          # zero-initialised prior
    c=RetrievalHead(8,['cf'],use_features=False)
    assert not c.needs_evidence and not c.has_fusion
