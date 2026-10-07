import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.models import ContextRestorer, RetrievalFusionExpert, BackboneAwareRestorer
from fare.structural.tmcp import recover_batch


def toy():
    B,T,N=2,4,5; adj=np.eye(N,dtype=np.float32); adj[0,1]=adj[1,0]=1
    x=torch.randn(B,T,N); obs=torch.ones(B,T,N,dtype=torch.bool); art=torch.zeros_like(obs); art[:,1,2]=True; obs[art]=False; x=x.masked_fill(~obs,0)
    tod=torch.arange(T)[None,:].expand(B,-1); dow=torch.zeros(B,T,dtype=torch.long)
    return x,obs,art,tod,dow,adj

def test_context_preserves_nonartificial():
    x,obs,art,tod,dow,adj=toy(); m=ContextRestorer(5,adj,32,2,4,0.0,2); y,h=m(x,obs,art,tod,dow)
    assert y.shape==x.shape and h.shape[:3]==x.shape
    assert torch.equal(y[~art],x[~art])

def test_retrieval_preserves_nonartificial():
    x,obs,art,tod,dow,adj=toy(); context=x.clone(); cand=torch.randn(2,3,4,5); cv=torch.ones_like(cand,dtype=torch.bool); sim=torch.randn(2,3)
    m=RetrievalFusionExpert(32,0.0); an=torch.tensor(adj); y,d=m(x,obs,art,context,cand,cv,sim,an)
    assert y.shape==x.shape and torch.equal(y[~art],x[~art]); assert d['weights'].shape==(2,3,5)

def test_backbone_aware_preserves_nonartificial():
    x,obs,art,tod,dow,adj=toy(); c=x.clone(); f=torch.randn(2,12,5); m=BackboneAwareRestorer(5,adj,12,32,2,4,0.0,2); y,d=m(x,obs,art,tod,dow,c,f)
    assert y.shape==x.shape and torch.equal(y[~art],x[~art])

def test_tmcp_preserves_observed():
    x=torch.rand(2,4,3,5); m=torch.rand_like(x)>0.2; sparse=torch.where(m,x,torch.zeros_like(x)); rec,info=recover_batch(sparse,m,max_iter=2)
    assert torch.allclose(rec[m],sparse[m]); assert rec.shape==x.shape
