import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.models import ContextRestorer, RetrievalFusionExpert, BackboneAwareRestorer
from fare.structural.tmcp import recover_batch
from fare.losses import baseline_relative_no_harm, pairwise_utility_ranking_loss, alpha_supervision_loss


def toy():
    B,T,N=2,4,5; adj=np.eye(N,dtype=np.float32); adj[0,1]=adj[1,0]=1
    x=torch.randn(B,T,N); obs=torch.ones(B,T,N,dtype=torch.bool); art=torch.zeros_like(obs); art[:,1,2]=True; obs[art]=False; x=x.masked_fill(~obs,0)
    tod=torch.arange(T)[None,:].expand(B,-1); dow=torch.zeros(B,T,dtype=torch.long)
    return x,obs,art,tod,dow,adj


def test_context_dense_estimate_and_control_preserve_nonartificial():
    x,obs,art,tod,dow,adj=toy(); m=ContextRestorer(5,adj,32,2,4,0.0,2,-3.0); y,d=m(x,obs,art,tod,dow)
    assert y.shape==x.shape and d['estimate'].shape==x.shape and d['gate'].shape==x.shape
    assert torch.equal(y[~art],x[~art]) and torch.equal(d['raw_restored'][~art],x[~art])


def test_retrieval_dense_estimate_and_two_gates():
    x,obs,art,tod,dow,adj=toy(); context=x.clone(); cand=torch.randn(2,3,4,5); cv=torch.ones_like(cand,dtype=torch.bool); sim=torch.randn(2,3)
    m=RetrievalFusionExpert(32,0.0,-1.5,-3.0); an=torch.tensor(adj); y,d=m(x,obs,art,context,cand,cv,sim,an)
    assert y.shape==x.shape and torch.equal(y[~art],x[~art]); assert d['weights'].shape==(2,3,5); assert d['query_candidate_scores'].shape==(2,3)
    assert d['support_gate'].shape==x.shape and d['control_gate'].shape==x.shape


def test_backbone_aware_dense_estimate_and_control():
    x,obs,art,tod,dow,adj=toy(); c=x.clone(); f=torch.randn(2,12,5); m=BackboneAwareRestorer(5,adj,12,32,2,4,0.0,2,-3.0); y,d=m(x,obs,art,tod,dow,c,f)
    assert y.shape==x.shape and torch.equal(y[~art],x[~art]); assert d['estimate'].shape==x.shape and d['gate'].shape==x.shape


def test_tmcp_preserves_observed():
    x=torch.rand(2,4,3,5); m=torch.rand_like(x)>0.2; sparse=torch.where(m,x,torch.zeros_like(x)); rec,info=recover_batch(sparse,m,max_iter=2)
    assert torch.allclose(rec[m],sparse[m]); assert rec.shape==x.shape


def test_no_harm_loss_positive_when_expert_worse():
    y=torch.zeros(2,2,2); mask=torch.ones_like(y,dtype=torch.bool); base=torch.zeros_like(y); expert=torch.ones_like(y)
    p,em,bm=baseline_relative_no_harm(expert,base,y,mask)
    assert float(p)>0 and float(em)>float(bm)


def test_pairwise_ranking_prefers_correct_order_and_beats_random_baseline():
    u=torch.tensor([[0.001,0.006,0.003,-0.002]])
    good=torch.tensor([[0.,3.,2.,-1.]],requires_grad=True)
    bad=torch.tensor([[3.,0.,1.,2.]],requires_grad=True)
    lg,ag,cg=pairwise_utility_ranking_loss(good,u,min_gap=1e-5)
    lb,ab,cb=pairwise_utility_ranking_loss(bad,u,min_gap=1e-5)
    assert float(lg.detach()) < float(lb.detach())
    assert float(ag.detach()) > float(ab.detach()) and float(cg)==float(cb)>0


def test_alpha_supervision_targets_query_mean_gate():
    _,_,art,_,_,_=toy(); gate=torch.full(art.shape,0.5,requires_grad=True); target=torch.tensor([0.5,0.25])
    loss,mae=alpha_supervision_loss(gate,art,target)
    assert float(loss.detach())>0 and float(mae.detach())>0
