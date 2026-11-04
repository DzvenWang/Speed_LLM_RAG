"""v7.1 C1 utility, set-complementarity, numerical safety and identity tests."""
from __future__ import annotations
import torch,pytest,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tcrr_stla.v7.losses import full_activation_targets,ranking_targets,cfrm_auxiliary,masked_mean
from tcrr_stla.v7.transport import SupportConditionedTransport
from tcrr_stla.v7.version import VERSION,RUN_NAMESPACE


def fake(B=2,K=4,P=3,N=5):
    torch.manual_seed(9)
    out=dict(base_prediction=torch.randn(B,P,N),
        normal=torch.randn(B,P,N),
        candidate_residuals=torch.randn(B,K,P,N)*.2,
        candidate_available=torch.ones(B,K,P,N,dtype=torch.bool),
        residual_bound=3.,
        candidate_transfer_gain=torch.ones(B,K,P,N,requires_grad=True)*.2,
        candidate_transfer_harm=torch.ones(B,K,P,N,requires_grad=True)*.2,
        candidate_logits=torch.randn(B,K+1,P,N,requires_grad=True),
        normal_component=torch.randn(B,K,P,N)*.2,
        cf_increment=torch.randn(B,K,P,N)*.1,
        cf_incremental_utility=torch.randn(B,K,P,N,requires_grad=True),
        cf_gain_logit=torch.randn(B,K,P,N,requires_grad=True),
        candidate_loo_proposal=torch.randn(B,K,P,N)*.2,
        marginal_reference_proposal=torch.randn(B,P,N)*.2,
        candidate_marginal_utility_pred=torch.randn(B,K,P,N,requires_grad=True))
    batch=dict(y_norm=torch.randn(B,P,N),future_valid=torch.ones(B,P,N,dtype=torch.bool))
    return out,batch


def test_full_activation_utility_uses_normal_not_backbone():
    out,b=fake(B=1,K=2,P=1,N=1)
    out['base_prediction']=torch.tensor([[[10.]]])
    out['normal']=torch.tensor([[[9.]]])
    out['candidate_residuals']=torch.zeros(1,2,1,1)
    b['y_norm']=torch.tensor([[[9.]]])
    u,none,valid=full_activation_targets(out,b)
    # Full candidate proposal = P0 (10) is worse than Normal (9).
    torch.testing.assert_close(u,torch.full_like(u,-1.))
    torch.testing.assert_close(none,torch.tensor([[[-1.]]]))
    assert valid.all()


def test_none_has_real_utility_and_ranking_temperature_is_active():
    u=torch.tensor([[[[1.0]]],[[[0.0]]]]).transpose(0,1)  # [1,2,1,1]
    v=torch.ones_like(u,dtype=torch.bool)
    none=torch.tensor([[[3.]]])
    p1=ranking_targets(u,none,v,.25)
    p2=ranking_targets(u,none,v,2.)
    assert p1.shape==(1,3,1,1)
    assert p1[0,-1,0,0]>p2[0,-1,0,0]
    assert p1[0,-1,0,0]>.99
    with pytest.raises(ValueError):ranking_targets(u,none,v,0)


def test_invalid_future_nan_and_all_invalid_candidates():
    out,b=fake()
    out['candidate_available'].zero_()
    b['future_valid'][:,-1,1]=False
    b['y_norm'][:,-1,1]=float('nan')
    u,none,valid=full_activation_targets(out,b)
    assert not valid.any()
    assert torch.isfinite(u).all() and torch.isfinite(none).all()
    aux=cfrm_auxiliary(out,b,0.,1.,target_temperature=.25)
    assert torch.isfinite(aux)
    aux.backward()
    assert out['candidate_logits'].grad is not None


def test_masked_mean_is_finite_with_nan_only_on_invalid_cells():
    x=torch.tensor([[float('nan'),2.],[4.,float('inf')]])
    valid=torch.tensor([[False,True],[True,False]])
    torch.testing.assert_close(masked_mean(x,valid),torch.tensor(3.))
    with pytest.raises(RuntimeError):masked_mean(x,~valid)


def test_candidate_set_permutation_equivariance_and_all_invalid():
    B,K,P,N,L,D=2,4,3,5,8,6
    torch.manual_seed(34)
    tr=SupportConditionedTransport(P,L,D,hidden=16,prediction_temperature=1.)
    base=torch.randn(B,P,N)
    ctx=torch.randn(B,N,L)
    normal=torch.randn(B,K,P,N)
    cf=torch.randn_like(normal)*.05
    rel=torch.rand_like(normal)
    support={'embedding':torch.randn(B,K,N,D), 'compatibility_base':torch.rand(B,K,N)}
    score=torch.randn(B,K);cov=torch.rand(B,K)
    valid=torch.ones(B,K,P,N,dtype=torch.bool)
    valid[:,:,:,0]=False
    kwargs=dict(base=base,context=ctx,normal=normal,cf_increment=cf,cf_reliability=rel,
                support=support,score=score,coverage=cov,valid=valid,cf_utility=cf*.3)
    a=tr(**kwargs)
    perm=torch.tensor([3,1,0,2])
    pr=dict(kwargs)
    for k in ('normal','cf_increment','cf_reliability','score','coverage','valid','cf_utility'):
        pr[k]=kwargs[k][:,perm]
    pr['support']={k:v[:,perm] for k,v in support.items()}
    b=tr(**pr)
    torch.testing.assert_close(a['proposed_correction'],b['proposed_correction'],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(a['candidate_weights'][:,perm],b['candidate_weights'][:,:K],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(a['candidate_weights'][:,-1],b['candidate_weights'][:,-1],atol=1e-5,rtol=1e-5)
    assert a['candidate_weights'][:,-1,:,0].min()>.99999
    assert torch.isfinite(a['set_disagreement']).all()
    assert torch.isfinite(a['set_direction_agreement']).all()
    # Temperature must affect actual prediction weights, not just config parsing.
    tr.prediction_temperature=2.
    c=tr(**kwargs)
    assert not torch.allclose(a['candidate_weights'],c['candidate_weights'])


def test_model_supports_k1_and_no_feature_quadratic_memory():
    tr=SupportConditionedTransport(3,8,6,hidden=16)
    assert tr.marginal_head[0].in_features==16+9
    assert VERSION=='7.1.0' and RUN_NAMESPACE=='v7_1_0_c1'
    p=Path(ROOT/'tcrr_stla/v7/transport.py').read_text()
    assert '[B,K,K,P,N]' in p
