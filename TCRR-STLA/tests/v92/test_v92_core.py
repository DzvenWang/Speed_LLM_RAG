import torch
import numpy as np
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v92.model import TCRRV92
from tcrr_stla.v92.training import configure_stage
from tcrr_stla.v92.utility import cf_training_loss,oracle_alignment,marginal_targets


def batch(b=2,k=4,n=7,h=12):
    torch.manual_seed(7)
    return dict(base=torch.randn(b,h,n),context=torch.randn(b,n,8),
          x=torch.randn(b,12,n),mask=(torch.rand(b,12,n)>.3).float(),
          normal=torch.randn(b,k,h,n),cf=torch.randn(b,k,h,n),score=torch.randn(b,k),
          valid=torch.ones(b,k,dtype=torch.bool),match=torch.rand(b,k,n,2))

def model(n=7):
    f=torch.eye(n).roll(1,0)
    return TCRRV92(GraphBundle(f),8,hidden=16,cf_hidden=8)

def test_cf_zero_invariant():
    m=model();b=batch();b['cf'].zero_()
    o=m(**b)
    assert torch.equal(o['prediction'],o['normal_prediction'])
    assert torch.equal(o['cf_increment'],torch.zeros_like(o['cf_increment']))

def test_response_sign_and_order_affect_utility():
    m=model();b=batch()
    u=m(**b)['cf_utility'];v=m(**{**b,'cf':-b['cf']})['cf_utility']
    t=m(**{**b,'cf':b['cf'].flip(2)})['cf_utility']
    assert (u-v).abs().max()>1e-5
    assert (u-t).abs().max()>1e-5

def test_gradients_and_normal_freeze():
    m=model();b=batch();y=torch.randn_like(b['base']);valid=torch.ones_like(y,dtype=torch.bool)
    before={n:p.detach().clone() for n,p in m.normal_core.named_parameters()}
    cfg=dict(utility_weight=.02,rank_weight=.01,marginal_weight=.005,alignment_weight=.002,full_info_weight=.01)
    for stage in ('align','utility','joint','paired'):
        configure_stage(m,stage)
        o=m(**b,rank_mode='uniform' if stage=='align' else 'learned')
        loss,_=cf_training_loss(o,y,valid,b['valid'],cfg,stage,teacher_cf=torch.randn_like(y),raw_cf=b['cf'])
        loss.backward()
        for n,p in m.normal_core.named_parameters():
            assert not p.requires_grad and p.grad is None
            assert torch.equal(p.detach(),before[n])
        if stage in ('align','joint','paired'):
            assert m.alignment.weight.grad is not None
        m.zero_grad(set_to_none=True)

def test_invalid_future_never_affects_teacher():
    m=model();b=batch();o=m(**b)
    y=torch.randn_like(b['base']);valid=torch.zeros_like(y,dtype=torch.bool)
    valid[:,0]=True
    from tcrr_stla.v9.model import utility_targets
    before=utility_targets(o['base'],o['normal_raw'],o['candidate_cf'],y,b['valid'],m.bound,3,valid)
    y[:,1:]=10000
    after=utility_targets(o['base'],o['normal_raw'],o['candidate_cf'],y,b['valid'],m.bound,3,valid)
    assert torch.equal(before,after)

def test_oracle_marginal_and_null():
    m=model();b=batch();o=m(**b);y=torch.randn_like(o['prediction']);valid=torch.ones_like(y,dtype=torch.bool)
    u,uv=marginal_targets(o,y,valid,b['valid'])
    assert u.shape==(2,4,3,7) and uv.all()
    alpha,support,details=oracle_alignment(o,b['cf'],y,valid,b['valid'])
    assert alpha.shape==(2,4,3,7) and support.shape==alpha.shape
    assert torch.isfinite(u).all()
    b['valid'].zero_()
    o=m(**b)
    assert torch.equal(o['prediction'],o['normal_prediction'])
    assert torch.allclose(o['cf_weights'][:,-1],torch.ones_like(o['cf_weights'][:,-1]))

def test_graph_modes_and_no_graph_fallback():
    b=batch();m=model()
    for mode in ('none','fixed','gnn','gnn_reliability'):
        o=m(**b,graph_mode=mode)
        assert o['cf_graph_features'].shape==(2,4,3,7,8)
        assert torch.isfinite(o['prediction']).all()
    assert torch.count_nonzero(m(**b,graph_mode='none')['cf_graph_features'])==0

def test_query_visibility_no_hidden_future_in_forward():
    m=model();b=batch()
    assert 'y' not in b and 'x_full' not in b
    out=m(**b)
    assert set(['cf_utility','alignment','cf_weights','graph_gates','normal_prediction'])<=set(out)

def test_recoverability_head_actually_changes_selection():
    m=model();b=batch()
    with torch.no_grad():
        m.recoverability_scale.fill_(2.)
        m.full_info_head.weight.fill_(.25)
        m.full_info_head.bias.fill_(.4)
    good=m(**b,recoverability_enabled=True)
    off=m(**b,recoverability_enabled=False)
    assert (good['cf_weights']-off['cf_weights']).abs().max()>1e-6


def test_no_graph_messages_without_neighbours():
    zero=GraphBundle(torch.zeros(7,7))
    m=TCRRV92(zero,8,hidden=16,cf_hidden=8)
    out=m(**batch(),graph_mode='gnn_reliability')
    assert torch.count_nonzero(out['cf_graph_features'])==0
    assert torch.count_nonzero(out['graph_gates'])==0
