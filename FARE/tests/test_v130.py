"""Regression tests for FARE v1.3 observability, ranking and legal fixed-pool probes."""
import sys,copy
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_full import fixture
from fare.ranking import target_recovery_ranking_loss,target_candidate_errors
from fare.losses import losses
from diagnose_fixed_pool import fixed_pool_weights

def _input(batch):
    return {k:batch[k] for k in ('x_obs','obs_mask','tod','dow','window_start')}

def test_observability_reaches_query_recovery_and_control():
    cfg,m,mem,b=fixture(n=12,b=2,m=40)
    m.set_phase('full');m.train();m.refresh(mem)
    out=m(_input(b),mem)
    assert out['observability_embedding'].shape==(2,m.regions.g,64)
    assert torch.isfinite(out['prediction']).all()
    loss=losses(m,b,out,mem,'full',0.,1.,cfg)['total']
    loss.backward(retain_graph=True)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.observability.parameters()),'Observability encoder is disconnected'
    assert m.retrieval.encoder.observability_query.weight.grad is not None
    assert m.retrieval.encoder.observability_query.weight.grad.abs().sum()>0
    assert m.state_recovery.obs_health.weight.grad is not None
    assert m.state_recovery.obs_health.weight.grad.abs().sum()>0
    # Controller's zero-initialized projection legitimately blocks forecast
    # gradients to reliability on the FIRST update. Verify the conf pathway
    # is differentiable, then assert gradient to its observability adapter.
    out['retrieval_confidence'].mean().backward()
    assert m.reliability.obs_reliability.weight.grad is not None
    assert m.reliability.obs_reliability.weight.grad.abs().sum()>0
    print('FARE_OBSERVABILITY_GRADIENT_PASS',flush=True)

def test_rank_diagnostic_normalized_and_missing_labels():
    cfg,m,mem,b=fixture(n=12,b=2,m=45)
    b['artificial_mask']=~b['obs_mask'];b['raw_valid']=torch.ones_like(b['obs_mask'])
    out=m(_input(b),mem)
    ret={'ids':out['retrieved_ids'],'weights':out['retrieved_weights'],'valid':out['retrieved_valid_candidates']}
    loss,diag=target_recovery_ranking_loss(b,ret,mem,m.regions)
    assert torch.isfinite(loss)
    assert 'ranking_target_valid_weight_mass' in diag
    # Manufacture a valid candidate without any target label: one region's
    # last candidate is excluded from the label set, not retrieval itself.
    old=mem.valid.clone();mem.valid[ret['ids'][0,0,-1]]=False
    loss,diag=target_recovery_ranking_loss(b,ret,mem,m.regions)
    assert torch.isfinite(loss)
    assert 0<=float(diag['ranking_target_valid_weight_mass'])<=1.000001
    err,labels=target_candidate_errors(b,ret,mem,m.regions)
    eligible=labels.sum(-1)>=2
    if eligible.any():
        w=ret['weights']*labels.float();w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        expected=((err*w).sum(-1)[eligible]).mean()
        assert torch.allclose(diag['ranking_selected_error'],expected,atol=1e-5)
    mem.valid.copy_(old)
    print('FARE_RANKING_NORMALIZATION_PASS',flush=True)

def test_fixed_pool_weights_and_overlap_guard():
    cfg,m,mem,b=fixture(n=12,b=2,m=45);m.eval()
    inp=_input(b)
    with torch.no_grad():
        ctx=m.regions(inp['x_obs'],inp['obs_mask'])
        ret=m.retrieval(ctx,m.regions,mem,inp['window_start'],observability=m.observability(ctx))
        err,labels=target_candidate_errors(b,ret,mem,m.regions)
        for variant in ('learned','uniform','state_oracle','onehot_0'):
            weights=fixed_pool_weights(ret,err,labels,variant)
            assert weights.shape==ret['weights'].shape
            assert torch.isfinite(weights).all()
            assert torch.allclose(weights.sum(-1)[ret['group_valid']],torch.ones_like(weights.sum(-1)[ret['group_valid']]),atol=1e-5)
            out=m(inp,mem,retrieval_override={**ret,'weights':weights})
            assert torch.isfinite(out['prediction']).all()
        tampered=dict(ret)
        tampered['ids']=ret['ids'].clone()
        # Force exactly overlapping candidate with an otherwise true validity flag.
        mem.starts[tampered['ids'][0,0,0]]=b['window_start'][0]
        try:m(inp,mem,retrieval_override=tampered)
        except ValueError as exc:assert 'overlapping' in str(exc)
        else:raise AssertionError('Overlapping memory was accepted')
    print('FARE_FIXED_POOL_LEAKAGE_GUARD_PASS',flush=True)

def test_inference_has_no_query_truth():
    cfg,m,mem,b=fixture(n=15,b=2,m=45)
    m.eval()
    with torch.no_grad():
        inp=_input(b)
        out=m(inp,mem)
        polluted=copy.deepcopy(b);polluted['x_full']=torch.randn_like(b['x_full'])*1000
        polluted['y_raw']=torch.randn_like(b['y_raw'])*1000
        out2=m(_input(polluted),mem)
        assert torch.allclose(out['prediction'],out2['prediction'])
    print('FARE_INFERENCE_NO_LABELS_PASS',flush=True)
if __name__=='__main__':
    for fn in (test_observability_reaches_query_recovery_and_control,
               test_rank_diagnostic_normalized_and_missing_labels,
               test_fixed_pool_weights_and_overlap_guard,
               test_inference_has_no_query_truth):
        fn()
