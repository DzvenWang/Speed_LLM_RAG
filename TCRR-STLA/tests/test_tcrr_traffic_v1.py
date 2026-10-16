"""Self-contained CPU unit tests for v1 traffic-conditioned retrieval."""
import tempfile
from pathlib import Path
import torch
from tcrr_stla.traffic import node_features,global_signature,TrafficResidualHead
from tcrr_stla.traffic_index import traffic_topk


def test_features_all_missing_finite():
    x=torch.randn(2,12,6)
    m=torch.zeros_like(x)
    f=node_features(x,m)
    assert f.shape==(2,6,8) and torch.isfinite(f).all()
    assert (f[...,4]==0).all()
    assert torch.isfinite(global_signature(x,m)).all()


def test_train_only_temporal_exclusion():
    x=torch.rand(2,12,6);m=torch.ones_like(x)
    bank=torch.rand(30,12)
    starts=torch.arange(30)*30
    queries=torch.tensor([0,60])
    ids,_=traffic_topk(x,m,bank,starts,queries,24,4)
    assert all(abs(int(starts[k])-int(queries[b]))>=24 for b,row in enumerate(ids) for k in row)


def test_head_has_gradients():
    head=TrafficResidualHead()
    x=torch.randn(2,12,5);m=torch.ones_like(x)
    c=torch.randn(2,3,12,5)
    out=head(x,m,c)
    (out['expected_gain'].square().mean()+out['logit_bias'].mean()).backward()
    assert head.match[-1].weight.grad is not None
