import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.models import ContextRestorer, RetrievalFusionExpert, BackboneAwareRestorer, HierarchicalInterventionController
from fare.structural.tmcp import recover_batch
from fare.losses import (
    baseline_relative_no_harm, pairwise_utility_ranking_loss,
    intervention_landscape_loss, reliability_supervision_loss,
)

ALPHA=[0.0,0.25,0.5,0.75,1.0]

def toy():
    B,T,N=2,4,5; adj=np.eye(N,dtype=np.float32); adj[0,1]=adj[1,0]=1
    x=torch.randn(B,T,N); obs=torch.ones(B,T,N,dtype=torch.bool); art=torch.zeros_like(obs); art[:,1,2]=True; obs[art]=False; x=x.masked_fill(~obs,0)
    tod=torch.arange(T)[None,:].expand(B,-1); dow=torch.zeros(B,T,dtype=torch.long)
    return x,obs,art,tod,dow,adj


def assert_ctrl(d,art):
    assert d['landscape'].shape==(art.shape[0],len(ALPHA))
    assert d['selected_alpha'].shape==(art.shape[0],)
    assert d['reliability'].shape==art.shape
    assert d['gate'].shape==art.shape
    assert torch.all(d['gate']>=0) and torch.all(d['gate']<=1)


def test_context_landscape_reliability_preserve_nonartificial():
    x,obs,art,tod,dow,adj=toy(); m=ContextRestorer(5,adj,32,2,4,0.0,2,ALPHA); y,d=m(x,obs,art,tod,dow)
    assert y.shape==x.shape and d['estimate'].shape==x.shape
    assert torch.equal(y[~art],x[~art]) and torch.equal(d['raw_restored'][~art],x[~art]); assert_ctrl(d,art)


def test_retrieval_landscape_and_ranker():
    x,obs,art,tod,dow,adj=toy(); context=x.clone(); cand=torch.randn(2,3,4,5); cv=torch.ones_like(cand,dtype=torch.bool); sim=torch.randn(2,3)
    m=RetrievalFusionExpert(32,0.0,-1.5,ALPHA); an=torch.tensor(adj); y,d=m(x,obs,art,context,cand,cv,sim,an)
    assert y.shape==x.shape and torch.equal(y[~art],x[~art]); assert d['weights'].shape==(2,3,5); assert d['query_candidate_scores'].shape==(2,3)
    assert d['support_gate'].shape==x.shape; assert_ctrl(d,art)


def test_backbone_aware_landscape():
    x,obs,art,tod,dow,adj=toy(); c=x.clone(); f=torch.randn(2,12,5); m=BackboneAwareRestorer(5,adj,12,32,2,4,0.0,2,ALPHA); y,d=m(x,obs,art,tod,dow,c,f)
    assert y.shape==x.shape and torch.equal(y[~art],x[~art]); assert d['estimate'].shape==x.shape; assert_ctrl(d,art)


def test_alpha_override_preserves_mean_intervention_without_clipping():
    x,obs,art,tod,dow,adj=toy(); m=ContextRestorer(5,adj,32,2,4,0.0,2,ALPHA)
    alpha=torch.tensor([0.25,0.5]); _,d=m(x,obs,art,tod,dow,alpha_override=alpha)
    for q in range(2):
        got=float(d['gate'][q][art[q]].mean().detach())
        assert abs(got-float(alpha[q])) < 1e-4


def test_tmcp_preserves_observed():
    x=torch.rand(2,4,3,5); m=torch.rand_like(x)>0.2; sparse=torch.where(m,x,torch.zeros_like(x)); rec,info=recover_batch(sparse,m,max_iter=2)
    assert torch.allclose(rec[m],sparse[m]); assert rec.shape==x.shape


def test_no_harm_loss_positive_when_expert_worse():
    y=torch.zeros(2,2,2); mask=torch.ones_like(y,dtype=torch.bool); base=torch.zeros_like(y); expert=torch.ones_like(y)
    p,em,bm=baseline_relative_no_harm(expert,base,y,mask)
    assert float(p)>0 and float(em)>float(bm)


def test_pairwise_ranking_prefers_correct_order_and_beats_random_baseline():
    u=torch.tensor([[0.001,0.006,0.003,-0.002]])
    good=torch.tensor([[0.,3.,2.,-1.]],requires_grad=True); bad=torch.tensor([[3.,0.,1.,2.]],requires_grad=True)
    lg,ag,cg=pairwise_utility_ranking_loss(good,u,min_gap=1e-5); lb,ab,cb=pairwise_utility_ranking_loss(bad,u,min_gap=1e-5)
    assert float(lg.detach()) < float(lb.detach()); assert float(ag.detach()) > float(ab.detach()) and float(cg)==float(cb)>0


def test_intervention_landscape_loss_recovers_best_alpha():
    target=torch.tensor([[0.0,0.01,0.03,0.02,-0.01],[0.0,-0.01,-0.02,-0.03,-0.04]])
    pred=target*100.0
    out=intervention_landscape_loss(pred,target,torch.tensor(ALPHA),100.0,1e-4,1.0)
    assert float(out['alpha_acc'])==1.0
    assert float(out['alpha_mae'])==0.0
    assert float(out['curve_mae_pp']) < 1e-6


def test_reliability_supervision_rewards_low_error_cells():
    est=torch.tensor([[[0.,2.,0.5]]]); full=torch.tensor([[[0.,0.,0.]]]); art=torch.ones_like(est,dtype=torch.bool)
    rel=torch.tensor([[[1.0,0.1,0.5]]],requires_grad=True)
    loss,mae,target=reliability_supervision_loss(rel,est,full,art)
    assert target[0,0,0] > target[0,0,2] > target[0,0,1]
    assert torch.isfinite(loss) and torch.isfinite(mae)
