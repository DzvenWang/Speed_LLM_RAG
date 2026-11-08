import numpy as np
import pytest
import torch
from tcrr_stla.v9.graph import GraphBundle,index_to_adj
from tcrr_stla.v9.model import TCRRV9,train_loss,paired_cross_failure_loss
from tcrr_stla.v9.views import make_mask,paired_mask,mask_sha
from tcrr_stla.v9.prepare import time_legal_ids


def fake(seed=10,b=2,k=4,h=12,n=6,c=8):
    torch.manual_seed(seed)
    a=index_to_adj(torch.tensor([[1,2],[0,2],[0,1],[1,2],[3,5],[3,4]]),n)
    graph=GraphBundle(a,a,a)
    model=TCRRV9(graph,c,hidden=12,blocks=3)
    kwargs=dict(base=torch.randn(b,h,n),context=torch.randn(b,n,c),
                x=torch.randn(b,12,n),mask=torch.randint(0,2,(b,12,n)).float(),
                normal=torch.randn(b,k,h,n)*.12,cf=torch.randn(b,k,h,n)*.07,
                score=torch.rand(b,k),valid=torch.ones(b,k,dtype=torch.bool),
                match=torch.rand(b,k,n,2))
    return model,kwargs

def test_zero_cf_exact_invariance():
    model,kw=fake()
    ref=model(**kw,enable_cf=False)
    o=model(**{**kw,'cf':torch.zeros_like(kw['cf'])})
    torch.testing.assert_close(ref['prediction'],o['prediction'],atol=0,rtol=0)
    torch.testing.assert_close(o['cf_increment'],torch.zeros_like(o['cf_increment']),atol=0,rtol=0)

def test_normal_anchor_invariant_to_cf_and_toggle():
    model,kw=fake()
    a=model(**kw,enable_cf=True)
    b=model(**{**kw,'cf':kw['cf']*100},enable_cf=True)
    c=model(**kw,enable_cf=False)
    torch.testing.assert_close(a['normal_prediction'],b['normal_prediction'],rtol=0,atol=0)
    torch.testing.assert_close(c['prediction'],a['normal_prediction'],rtol=0,atol=0)
    torch.testing.assert_close(a['normal_prediction']+a['cf_increment'],a['prediction'])

def test_all_invalid_selects_null():
    m,kw=fake();kw['valid']=torch.zeros_like(kw['valid'])
    o=m(**kw)
    torch.testing.assert_close(o['prediction'],o['normal_prediction'],atol=1e-6,rtol=0)
    assert torch.all(o['cf_weights'][:,-1]>.999)

def test_gradient_and_shapes():
    m,kw=fake();o=m(**kw)
    y=torch.randn_like(kw['base']);future=torch.ones_like(y,dtype=torch.bool)
    loss,_=train_loss(o,y,future,kw['valid'])
    assert torch.isfinite(loss)
    loss.backward()
    assert m.utility.weight.grad is not None
    assert m.amplitude.weight.grad is not None
    assert o['cf_utility'].shape==(2,4,3,6)
    assert o['cf_weights'].shape==(2,5,3,6)

def test_cross_failure_identical_candidate_ids():
    m,kw=fake();oa=m(**kw);ob=m(**{**kw,'cf':kw['cf']*.2})
    ids=torch.tensor([[1,2,3,4],[5,6,7,8]])
    ids2=ids.flip(1)
    target=torch.randn_like(kw['base']);valid=torch.ones_like(target,dtype=torch.bool)
    loss=paired_cross_failure_loss(oa,ob,ids,ids2,target,target,
                                   kw['valid'],kw['valid'],valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.utility.weight.grad is not None

def test_source_mask_never_fabricates_data():
    source=np.ones((12,7),bool);source[:,0]=False
    for family in ('entry_random','sensor_outage','temporal_outage'):
        m=make_mask(source,family=family,severity=40,seed=50)
        assert not m[:,0].any()
        assert (m<=source).all()
        assert np.array_equal(m,make_mask(source,family=family,severity=40,seed=50))
    assert mask_sha(source)!=mask_sha(~source)
    with pytest.raises(ValueError):make_mask(source,family='directional_outage',severity=40,seed=10)

def test_temporal_legal_candidates_only():
    times=np.arange(0,12*30,30)
    ids=time_legal_ids(180,times,k=4,history=12,horizon=12)
    assert all(times[ids]+24<=180)
    first=time_legal_ids(0,times,k=4,history=12,horizon=12)
    assert np.array_equal(first,np.zeros(4,dtype='int32'))

def test_graph_masks_unobserved_values():
    n=5;a=torch.eye(n).roll(1,0)
    g=GraphBundle(a,a,a)
    mask=torch.zeros(2,12,n)
    x=torch.randn(2,12,n)
    zero=g.query_summary(x,mask)
    altered=g.query_summary(x*1000,mask)
    torch.testing.assert_close(zero,altered)

def test_sparse_response_aggregation_matches_dense():
    m,kw=fake()
    graph=m.graph
    inp=kw['cf']
    got=graph.cf_summary(inp)
    expected=torch.stack([torch.einsum('ij,bkhj->bkhi',a,inp)
          for a in (graph.functional_signed,graph.upstream,graph.downstream)],-1)
    torch.testing.assert_close(got,expected)

def test_student_cannot_accept_full_query_as_inference_input():
    m,kw=fake()
    with pytest.raises(TypeError):m(**kw,x_full_training_only=torch.ones_like(kw['x']))
