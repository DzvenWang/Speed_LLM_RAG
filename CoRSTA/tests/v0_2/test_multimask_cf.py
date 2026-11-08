"""CF error-contrast tests: real error changes rather than prediction difference alone."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from corsta_v2.counterfactual.error_contrast import (
    teacher_gain,candidate_gain,error_contrast_losses)
from corsta_v2.data.dataset import structured_mask,_rng
from test_v2_core import make_model,batch

torch.set_num_threads(1)


def test_signed_error_difference_has_no_extra_information():
    y=torch.tensor([[0.,10.]])
    pf=torch.tensor([[2.,12.]])
    pm=torch.tensor([[0.,14.]])
    # Signed error contrast is just masked prediction minus full prediction.
    assert torch.equal((pm-y)-(pf-y),pm-pf)
    # Absolute error difference is DIFFERENT and may be negative.
    u=teacher_gain(pm,pf,y)
    assert torch.equal(u,torch.tensor([[-2.,2.]]))


def test_error_ranking_depends_on_future_not_just_mask_severity():
    y=torch.tensor([[0.]])
    pf=torch.tensor([[2.]])
    pa=torch.tensor([[0.]])
    pb=torch.tensor([[4.]])
    ua=teacher_gain(pa,pf,y);ub=teacher_gain(pb,pf,y)
    assert ua.item()<0<ub.item()
    assert torch.equal(ua-ub,(pa-y).abs()-(pb-y).abs())
    assert torch.equal(candidate_gain(pa,pf,y),torch.tensor([[-2.]]))


def test_two_masks_of_same_valid_source_remain_distinct_and_legal():
    src=np.ones((12,9),dtype=bool)
    src[3,4]=False
    a=structured_mask(src,40,'regional',_rng(10,1,2,40,'regional'))
    b=structured_mask(src,40,'directional',_rng(10,1,2,40,'directional-contrast'))
    assert a.sum()==b.sum()
    assert not np.array_equal(a,b)
    assert not np.any(a&~src) and not np.any(b&~src)


def test_train_only_error_contrast_backprop_to_cf_and_action():
    m=make_model();obs,b=batch()
    m.set_stage('cf',use_graph=False)
    obs2={**obs,'mask':obs['mask'].clone(),'x':obs['x'].clone()}
    obs2['mask'][:,0]=0
    obs2['x'][:,0]=0
    out=m(obs,enable_residual=False,enable_controller=False)
    other=m(obs2,enable_residual=False,enable_controller=False)
    from corsta_v2.counterfactual.teacher import PairedTeacher
    teacher=PairedTeacher(m.stla)(b,out['h0'],out['backbone'])
    cfg={'training':{'cf_error_gain_weight':0.04,'cf_error_rank_weight':0.02,
                      'cf_contrast_response_weight':0.03,'cf_acceptance_weight':0.02,
                      'cf_error_temperature':0.35}}
    loss,diagnostics=error_contrast_losses(out,other,b,teacher,cfg)
    assert torch.isfinite(loss)
    assert 'contrast_mask_a_gain' in diagnostics
    loss.backward()
    assert m.cf.error_gain_head[-1].weight.grad is not None
    assert m.cf.error_gain_head[-1].weight.grad.abs().sum()>0
    assert m.cf.acceptance_head[-1].weight.grad is not None
    assert m.cf.acceptance_head[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.stla.parameters())


def test_action_gate_changes_forecast_but_not_cf_candidate_target():
    m=make_model();m.eval();obs,_=batch()
    with torch.no_grad():
        out=m(obs,enable_residual=False,enable_controller=False)
        assert torch.allclose(out['cf_prediction'],out['backbone']+
                              out['cf_accept_gate']*out['delta_cf_candidate'])
        assert torch.allclose(out['cf_raw_prediction'],out['backbone']+
                              out['delta_cf_candidate'])
        m.cf.acceptance_head[-1].bias.fill_(-16.)
        suppressed=m(obs,enable_residual=False,enable_controller=False)
    assert torch.max((suppressed['cf_prediction']-suppressed['backbone']).abs())<1e-4
    assert torch.allclose(suppressed['delta_cf_candidate'],out['delta_cf_candidate'])


def test_no_future_in_masked_view_model_input():
    m=make_model();obs,b=batch()
    with torch.no_grad():
        a=m(obs,enable_residual=False)['cf_teacher_gain']
        wrong={**obs,'y_norm':b['y_norm']}
        try:
            m(wrong,enable_residual=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Model consumed privileged future')
        c=m(obs,enable_residual=False)['cf_teacher_gain']
    assert torch.equal(a,c)


def test_rejected_cf_does_not_reappear_through_residual_reference():
    """At zero residual shift, even a rejected CF must stay rejected."""
    m=make_model();m.eval();obs,_=batch()
    with torch.no_grad():
        m.cf.acceptance_head[-1].bias.fill_(-20.)
        # Force exactly zero residual latent shift at all masked queries.
        m.residual.head[-1].weight.zero_()
        m.residual.head[-1].bias.zero_()
        out=m(obs,enable_residual=True,enable_controller=False)
        assert torch.allclose(out['cf_residual_prediction'],out['cf_prediction'],atol=1e-6)
        assert torch.allclose(out['delta_residual'],torch.zeros_like(out['delta_residual']),atol=1e-6)
