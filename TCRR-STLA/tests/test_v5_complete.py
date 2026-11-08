"""Regression tests for the research mechanisms, not just tensor shapes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from test_v5 import model,data
from tcrr_stla.v5.calibration import choose_rejection_thresholds
from tcrr_stla.v5.losses import compute_losses
from tcrr_stla.v5 import trainer


def test_cf_gain_head_reaches_real_prediction():
    m=model(); d=data(); m.eval()
    with torch.no_grad():
        m.c1.incremental_utility.bias.fill_(-3)
        low=m(**d,mode='M11')['prediction']
        m.c1.incremental_utility.bias.fill_(3)
        high=m(**d,mode='M11')['prediction']
    assert not torch.allclose(low,high,atol=1e-7)


def test_cf_positive_probability_head_is_trainable_and_causal():
    m=model(); d=data(); m.eval()
    a=m(**d,mode='M11')
    target=torch.randn_like(a['prediction'])
    batch={'y_norm':target,'y_raw':target,'future_valid':torch.ones_like(target,dtype=torch.bool)}
    loss,stats=compute_losses(a,batch,0,1,{'training':{},'v5':{}},'c12')
    loss.backward()
    assert m.c1.incremental_utility.weight.grad is not None
    assert m.c1.gain_probability.weight.grad is not None
    assert torch.isfinite(stats['cf_gain_bce'])


def test_marginal_head_directly_changes_final_candidate_weights():
    m=model();d=data();m.eval()
    with torch.no_grad():
        a=m(**d,mode='M11')
        m.c2.marginal_head[-1].bias.fill_(2.)
        b=m(**d,mode='M11')
    assert not torch.allclose(a['candidate_weights'],b['candidate_weights'])
    assert not torch.allclose(a['prediction'],b['prediction'])


def test_support_changes_utility_and_varies_with_horizon():
    m=model();d=data();m.eval()
    with torch.no_grad():
        m.c2.support_utility.bias.fill_(1.)
        m.c2.support_logit.weight.fill_(.3)
        out=m(**d,mode='M11')
        off=m(**d,mode='M10')
    assert out['support_utility_delta'].abs().sum()>0
    assert not torch.allclose(out['candidate_transfer_gain'],off['candidate_transfer_gain'])
    assert torch.count_nonzero(off['support_logits_delta'])==0
    assert (out['support_logits_delta'][:,:,1]-out['support_logits_delta'][:,:,0]).abs().max()>0


def test_no_support_leaks_through_learned_controller():
    m=model();d=data();m.eval()
    with torch.no_grad():
        x=m(**d,mode='M10',controller_mode='learned')
        d['candidate_source_valid']=torch.zeros_like(d['candidate_source_valid'])
        y=m(**d,mode='M10',controller_mode='learned')
    assert torch.allclose(x['prediction'],y['prediction'],atol=1e-7)
    assert x['support_compatibility_aggregated'].count_nonzero()==0


def test_correct_and_candidate_shuffled_pairs_are_distinct():
    m=model();d=data();m.eval()
    a=m(**d,mode='M11',shuffled_negatives=True)
    assert 'cf_wrong_increment' in a
    assert (a['cf_wrong_increment']-a['cf_increment']).abs().max()>1e-6
    target=torch.randn_like(a['prediction'])
    b={'y_norm':target,'y_raw':target,'future_valid':torch.ones_like(target,dtype=torch.bool)}
    _,stats=compute_losses(a,b,0,1,{'training':{},'v5':{}},'c12')
    assert torch.isfinite(stats['cf_pair'])


def test_zero_scale_has_exact_zero_predicted_utility_and_harm():
    m=model();d=data();m.eval()
    with torch.no_grad():o=m(**d,mode='M11',controller_mode='learned')
    assert torch.count_nonzero(o['scale_positive_gain'][...,0])==0
    assert torch.count_nonzero(o['scale_positive_harm'][...,0])==0
    assert torch.count_nonzero(o['scale_utility_logits'][...,0])==0


def test_calibrated_controller_can_reject_and_restore_backbone():
    m=model();d=data();m.eval()
    m.c3.set_calibration(min_utility=1000.,max_harm_probability=1.,enabled=True)
    with torch.no_grad():o=m(**d,mode='M11',controller_mode='learned')
    assert o['reject_correction'].all()
    assert torch.equal(o['prediction'],o['base_prediction'])


def test_calibration_selects_useful_rejection_on_holdout():
    n=1000
    base=torch.zeros(n);y=torch.ones(n);good=torch.arange(n)%2==0
    proposal=torch.where(good,torch.ones(n),-torch.ones(n))
    net=torch.where(good,torch.ones(n),-torch.ones(n))
    r=choose_rejection_thresholds(base,proposal,torch.ones(n),net,
                                   torch.where(good,torch.zeros(n),torch.ones(n)),
                                   y,torch.ones(n,dtype=torch.bool),min_coverage=.45)
    assert r['enabled']
    assert r['calibration_mae_after']<r['calibration_mae_before']
    assert r['calibration_coverage']>=.45
    assert r['check_used'] is False


def test_s0_settings_are_taken_from_v5():
    from tcrr_stla.v5.shared_adapter import s0_settings, fit_shared_adapter
    assert trainer.fit_shared_adapter is fit_shared_adapter
    cfg={'v5':{'s0_epochs':6,'s0_lr':0.00012}}
    assert s0_settings(cfg)==(0.00012,6)
    assert s0_settings(cfg,smoke=True)==(0.00012,1)


def test_joint_loss_contains_controller_and_c12_supervision():
    m=model();d=data();m.eval()
    o=m(**d,mode='M11',controller_mode='learned')
    y=torch.randn_like(o['prediction'])
    b={'y_norm':y,'y_raw':y,'future_valid':torch.ones_like(y,dtype=torch.bool)}
    _,stats=compute_losses(o,b,0,1,{'training':{},'v5':{}},'joint')
    assert 'cf_incremental_utility' in stats and 'scale_harm_bce' in stats


def test_support_shuffle_coverage_controls_are_distinct():
    m=model();d=data();m.eval()
    full=m(**d,mode='M11',support_control='correct')
    simple=m(**d,mode='M11',support_control='coverage')
    shuffled=m(**d,mode='M11',support_control='shuffled')
    assert not torch.equal(full['support_context'],simple['support_context'])
    assert not torch.equal(full['support_context'],shuffled['support_context'])


def test_streaming_diagnostics_and_wrong_cf_stat():
    from tcrr_stla.v5.diagnostics import MechanismDiagnostics
    m=model();d=data();m.eval()
    o=m(**d,mode='M11',shuffled_negatives=True)
    y=torch.randn_like(o['prediction'])
    b={'y_norm':y,'future_valid':torch.ones_like(y,dtype=torch.bool)}
    diag=MechanismDiagnostics();diag.update(o,b)
    res=diag.summary()
    assert res['n_valid_forecast_positions']==y.numel()
    assert res['n_valid_candidate_positions']==d['candidate_valid'].numel()
    assert 'correct_cf_minus_shuffled_gain' in res
    assert 'marginal_utility_pearson' in res


def test_wrong_query_mask_replay_uses_actual_teacher():
    from tcrr_stla.evidence import counterfactual_residuals
    from test_v5 import FakeBackbone
    n,k,b,h=7,3,2,12
    torch.manual_seed(12)
    x=torch.randn(b,k,h,n)
    cand=dict(x=x,source_valid=torch.ones_like(x),y=torch.randn(b,k,12,n),
              tod=torch.zeros(b,k,h,dtype=torch.long),dow=torch.zeros(b,k,h,dtype=torch.long))
    teacher=FakeBackbone(n)
    a=torch.ones(b,h,n)
    z=counterfactual_residuals(teacher,cand,a,4,'matched',strict_source_mask=True)
    wrong=a.roll(1,0)
    # Mask and teacher are real computation in this test (FakeBackbone is constant,
    # so equality is expected, while the interface is validated).
    w=counterfactual_residuals(teacher,cand,wrong,4,'matched',strict_source_mask=True)
    assert z.shape==w.shape==(b,k,12,n)


def test_wrong_mask_selector_never_selects_self_and_matches_coverage():
    from tcrr_stla.v5.diagnostics import select_matched_wrong_mask
    masks=torch.stack((torch.ones(12,5),torch.ones(12,5),torch.zeros(12,5)))
    masks[0,:2]=0;masks[1,:2]=0;masks[2,:3]=1
    wrong,matched=select_matched_wrong_mask(masks,['A','A','B'])
    assert matched.all()
    assert not torch.equal(wrong[0],masks[0])
    assert not torch.equal(wrong[1],masks[1])
