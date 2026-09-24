#!/usr/bin/env python3
from __future__ import annotations
import sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.p1_memory import history_features,causal_topk,causal_random_k,aggregate_support
from presure.p1_model import P1ResidualAdapter


def main():
    rng=np.random.default_rng(10); Q,H,N,P,D,K=7,12,5,12,8,3
    x=rng.normal(size=(Q,H,N)).astype(np.float32); m=rng.random((Q,H,N))>0.35; x=np.where(m,x,0).astype(np.float32)
    hf=history_features(x,m); assert hf.shape==(Q,N,6); assert np.isfinite(hf).all()
    qk=rng.normal(size=(Q,D)).astype(np.float32);qk/=np.linalg.norm(qk,axis=1,keepdims=True);mk=qk.copy();qs=np.arange(Q)*40+100;ms=np.arange(Q)*40
    ii,ss,w=causal_topk(qk,mk,qs,ms,H,P,K,0.2,4)
    for q in range(Q):
        for j in ii[q][ii[q]>=0]: assert ms[j]+H+P < qs[q]
    ri,rw=causal_random_k(qs,ms,H,P,K,12)
    for q in range(Q):
        for j in ri[q][ri[q]>=0]: assert ms[j]+H+P < qs[q]
    fe=rng.normal(size=(Q,P,N)).astype(np.float32); fm=rng.random((Q,P,N))>.2
    ah,af,ac=aggregate_support(ii,w,hf,fe,fm); assert ah.shape==(Q,N,6) and af.shape==(Q,P,N) and ac.shape==(Q,P,N); assert np.isfinite(af).all()
    model=P1ResidualAdapter(D,6,P,64,0.0);base=torch.randn(2,P,N);latent=torch.randn(2,N,D);qh=torch.randn(2,N,6);sh=torch.randn(2,N,6);sf=torch.randn(2,P,N);sc=torch.rand(2,P,N)
    pred,delta=model(base,latent,qh,sh,sf,sc); assert torch.equal(pred,base); assert torch.count_nonzero(delta)==0
    print('SMOKE PreSuRe-P1 PASS: causal retrieval + observed-history features + masked episode aggregation + exact zero-start residual adapter',flush=True)
if __name__=='__main__':main()
