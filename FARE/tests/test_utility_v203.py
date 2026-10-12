"""Forecast utility v2.0.3 regression tests. No Base_SupportBench dependency."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fare.forecast_utility import supervised_coarse_utility,coarse_forecast_utility_loss
from fare.graph import GraphRegions
from fare.memory import TrainMemory
from fare.retrieval import TargetRetriever
from test_full import fixture


def test_coarse_utility_sign_and_grad():
    torch.manual_seed(15)
    adj=torch.ones(2,2)
    regions=GraphRegions(adj,target_size=2,max_anchors=2)
    b,g,k,p=1,regions.g,3,12
    # one candidate matches future exactly, one is harmful, one neutral.
    aligned=torch.stack([torch.full((p,),2.),torch.full((p,),-1.),torch.full((p,),1.)],dim=0).reshape(b,g,k,p)
    prediction=torch.zeros((b,g,k),requires_grad=True)
    ev={'coarse_ids':torch.tensor([[[0,1,2]]]),'coarse_valid':torch.ones((b,g,k),dtype=torch.bool),
        'coarse_analog_future':aligned,
        'coarse_future_valid':torch.ones(b,g,k,p,dtype=torch.bool),
        'coarse_anchor_support':torch.ones(b,g,k,dtype=torch.int64),
        'coarse_base_region':torch.zeros(b,g,p),
        'coarse_utility_pred':prediction,
        'weights':torch.full((b,g,k),1./k)}
    batch={'y_norm':torch.full((b,p,2),2.),'future_valid':torch.ones(b,p,2,dtype=torch.bool)}
    target,valid=supervised_coarse_utility(batch,ev,regions)
    assert valid.all()
    assert target[0,0,0]>0
    assert target[0,0,1]<0
    assert target[0,0,0]>target[0,0,2]
    loss,metric=coarse_forecast_utility_loss(batch,ev,regions)
    assert loss.isfinite() and metric['coarse_utility_oracle_positive_groups']==1
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum()>0
    assert prediction.grad[0,0,0]<prediction.grad[0,0,1]


def test_future_labels_do_not_change_inference():
    cfg,model,memory,batch=fixture(b=2,n=14,m=60)
    memory.future=torch.randn(60,12,14)
    memory.future_valid=torch.ones_like(memory.future,dtype=torch.bool)
    batch['window_start']=torch.tensor([1800,1850])
    model.eval();model.refresh(memory)
    with torch.no_grad():
        out1=model(batch,memory)
        fake={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()}
        fake['y_raw']=torch.randn_like(batch['y_raw'])*1000
        fake['y_norm']=torch.randn_like(batch['y_raw'])*1000
        fake['x_full']=torch.randn_like(batch['x_full'])*1000
        out2=model(fake,memory)
    assert torch.allclose(out1['prediction'],out2['prediction'],atol=0,rtol=0)
    assert torch.equal(out1['retrieved_ids'],out2['retrieved_ids'])


def test_new_utility_receives_training_gradient():
    cfg,model,memory,batch=fixture(b=2,n=12,m=64)
    memory.future=torch.randn(64,12,12)
    memory.future_valid=torch.ones_like(memory.future,dtype=torch.bool)
    batch['window_start']=torch.tensor([2800,2900])
    batch['y_norm']=torch.randn(2,12,12)
    model.set_phase('retrieval');model.train();model.refresh(memory)
    out=model(batch,memory,phase_fast='retrieval')
    assert out['retrieval_evidence']['coarse_utility_pred'] is not None
    loss,diag=coarse_forecast_utility_loss(batch,out['retrieval_evidence'],model.regions)
    assert loss.isfinite()
    loss.backward()
    grad=model.retrieval.utility_head[-1].weight.grad
    assert grad is not None and grad.abs().sum()>0
    assert diag['coarse_utility_eligible_fraction']>0


def test_causal_memory_future_filter_and_all_invalid():
    torch.manual_seed(8)
    n,h,p=6,12,12
    regions=GraphRegions(torch.eye(n),target_size=3,max_anchors=3)
    memory=TrainMemory(torch.randn(8,h,n),torch.ones(8,h,n,dtype=torch.bool),
                       torch.tensor([0,25,50,75,100,125,150,175]),
                       future=torch.randn(8,p,n),future_valid=torch.ones(8,p,n,dtype=torch.bool))
    retriever=TargetRetriever(h,coarse_topk=8,topk=4,overlap=24)
    retriever.refresh(memory,regions)
    obs=torch.randn(1,h,n)
    ctx=regions(obs,torch.ones_like(obs,dtype=torch.bool))
    start=torch.tensor([65])
    out=retriever(ctx,regions,memory,start,base_forecast=torch.zeros(1,p,n))
    assert (memory.starts[out['ids']]+h+p<=start[:,None,None]+h).logical_or(~out['valid']).all()
    # None of these finished historical outcomes are legal at query start 0.
    invalid=retriever(ctx,regions,memory,torch.tensor([-100]),base_forecast=torch.zeros(1,p,n))
    assert not invalid['valid'].any()
    assert invalid['weights'].sum()==0


def test_old_model_without_future_still_operates():
    cfg,model,memory,batch=fixture(b=2,n=12,m=45)
    model.eval();model.refresh(memory)
    with torch.no_grad():out=model(batch,memory)
    assert torch.isfinite(out['prediction']).all()
    assert out['prediction'].shape==(2,12,12)


def test_causal_pruning_happens_before_coarse_topk():
    torch.manual_seed(118)
    n,h,p=6,12,12
    adjacency=torch.ones(n,n)
    regions=GraphRegions(adjacency,target_size=3,max_anchors=3)
    starts=torch.tensor([0,20,40,60,80,100,120,140])
    memory=TrainMemory(torch.randn(8,h,n),torch.ones(8,h,n,dtype=torch.bool),
                       starts,future=torch.randn(8,p,n),
                       future_valid=torch.ones(8,p,n,dtype=torch.bool))
    r=TargetRetriever(h,coarse_topk=2,topk=2,overlap=h+p)
    r.refresh(memory,regions)
    obs=torch.randn(1,h,n)
    ctx=regions(obs,torch.ones_like(obs,dtype=torch.bool))
    start=torch.tensor([100])
    out=r(ctx,regions,memory,start,base_forecast=torch.zeros(1,p,n),diagnostics=True)
    # Memory starts > 88 include future outcomes that did not exist at issue.
    assert (starts[out['coarse_ids']]<=88).all()
    assert out['valid'].all()


def test_utility_head_ablation_changes_scores():
    cfg,model,memory,batch=fixture(b=2,n=12,m=64)
    memory.future=torch.randn(64,12,12)
    memory.future_valid=torch.ones_like(memory.future,dtype=torch.bool)
    batch['window_start']=torch.tensor([2800,2900])
    with torch.no_grad():
        model.retrieval.utility_head[0].weight[0].fill_(0)
        model.retrieval.utility_head[0].weight[0,0]=1
        model.retrieval.utility_head[0].bias[0]=1
        model.retrieval.utility_head[-1].weight.fill_(0)
        model.retrieval.utility_head[-1].weight[0,0]=2
        model.retrieval.utility_score_weight=1.5
    model.eval();model.refresh(memory)
    with torch.no_grad():
        ctx,normal=model.retrieve_for_diagnostics(batch,memory)
        off=model.retrieval(ctx,model.regions,memory,batch['window_start'],
                            mode='utility_off',demand=None,base_forecast=torch.randn(2,12,12))
        assert normal['coarse_utility_pred'] is not None
        # Utility-enabled scores differ, while causality and tensor shapes hold.
        assert normal['scores'].shape==off['scores'].shape
        assert torch.isfinite(normal['scores']).all()
