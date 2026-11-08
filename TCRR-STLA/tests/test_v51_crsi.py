"""Structural research tests for CFRM, SCRT and CRSI under shared weights."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from test_v5 import model, data
from tcrr_stla.v5.controller import VARIANTS
from tcrr_stla.v5.losses import compute_losses
from tcrr_stla.v5.diagnostics import MechanismDiagnostics


@pytest.mark.parametrize('name', ['M000', 'M001', 'M010', 'M011', 'M100', 'M101', 'M110', 'M111'])
def test_exact_nonlinear_component_decomposition_and_orthogonal_switches(name):
    m=model(); m.eval(); o=m(**data(),mode=name)
    assert torch.allclose(o['proposed_normal']+o['proposed_cf'],o['proposed_correction'],atol=1e-6)
    if name[1]=='0':
        assert o['proposed_cf'].count_nonzero()==0
    if name[3]=='0':
        assert torch.equal(o['prediction'],o['base_prediction']+o['proposed_correction'])
        assert torch.equal(o['normal_gate'],torch.ones_like(o['normal_gate']))
        assert torch.equal(o['cf_gate'],torch.ones_like(o['cf_gate']))
    else:
        assert torch.allclose(o['prediction'],o['base_prediction']+
            o['normal_gate']*o['proposed_normal']+o['cf_gate']*o['proposed_cf'],atol=1e-6)


def test_nonlinear_opposing_components_not_naive_tanh_sum():
    m=model(); d=data()
    with torch.no_grad():
        o=m(**d,mode='M110')
    limit=m.c2.bound
    naive=limit*torch.tanh(o['raw_normal']/limit)+limit*torch.tanh(o['raw_cf']/limit)
    assert not torch.allclose(o['proposed_correction'],naive,atol=1e-7)
    assert torch.allclose(o['proposed_normal']+o['proposed_cf'],o['proposed_correction'],atol=1e-6)


def test_component_control_can_keep_normal_and_reject_cf():
    m=model(); d=data();m.eval()
    with torch.no_grad():
        m.c3.base_logits.weight.zero_()
        m.c3.base_logits.bias.copy_(torch.tensor([9.,-9.]))
        out=m(**d,mode='M111',controller_variant='component')
    assert out['normal_gate'].mean()>.99
    assert out['cf_gate'].mean()<.01
    assert torch.allclose(out['prediction'],out['base_prediction']+out['proposed_normal'],atol=.01)


@pytest.mark.parametrize('variant', VARIANTS)
def test_controller_variants_are_operational_and_finite(variant):
    m=model();o=m(**data(),mode='M111',controller_variant=variant)
    assert o['controller_variant']==variant
    assert torch.isfinite(o['prediction']).all()
    assert (o['gate']>=0).all() and (o['gate']<=1).all()


def test_consistency_branch_changes_prediction_after_training_weights():
    m=model(); d=data(); m.eval()
    with torch.no_grad():
        m.c3.consistency_net[-1].weight.fill_(0.03)
        m.c3.consistency_net[-1].bias.copy_(torch.tensor([.2,-.2]))
        component=m(**d,mode='M111',controller_variant='component')['prediction']
        agreement=m(**d,mode='M111',controller_variant='agreement')['prediction']
    assert not torch.allclose(component,agreement,atol=1e-7)


def test_localization_branch_changes_prediction_when_weights_trained():
    m=model(n=13,p=7); d=data(n=13,p=7,k=3)
    m.neighbours=torch.arange(13)[:,None].roll(1,0)
    with torch.no_grad():
        m.c3.coarse_net[-1].weight.fill_(.015)
        m.c3.coarse_net[-1].bias.copy_(torch.tensor([.3,-.3]))
        component=m(**d,mode='M111',controller_variant='component')['prediction']
        localized=m(**d,mode='M111',controller_variant='localized')['prediction']
    assert not torch.allclose(component,localized,atol=1e-7)


def test_two_source_utility_and_harm_supervision_backpropagate():
    m=model();m.set_stage('controller');d=data()
    o=m(**d,mode='M111')
    target=torch.randn_like(o['prediction'])
    batch=dict(y_norm=target,y_raw=target,future_valid=torch.ones_like(target,dtype=torch.bool))
    loss,stats=compute_losses(o,batch,0.,1.,{'training':{},'v5':{}},stage='controller')
    loss.backward()
    assert torch.isfinite(loss)
    assert 'component_utility' in stats and 'component_harm' in stats
    assert m.c3.component_utility.weight.grad is not None
    assert m.c3.component_harm.weight.grad is not None


def test_component_diagnostics_report_harm_agreement_interaction():
    m=model();o=m(**data(),mode='M111')
    valid=torch.ones_like(o['prediction'],dtype=torch.bool)
    batch=dict(y_norm=torch.randn_like(o['prediction']),future_valid=valid)
    diag=MechanismDiagnostics();diag.update(o,batch)
    summary=diag.summary()
    for key in ('normal_component_gain','cf_component_gain','component_interaction_gain',
                'component_conflict','normal_agreement','cf_agreement',
                'normal_intervention_strength','cf_intervention_strength'):
        assert key in summary and summary[key] is not None


def test_large_net_shape_without_hardcoding_node_count():
    m=model(n=388);m.eval(); d=data(n=388,k=12,b=1)
    with torch.no_grad():o=m(**d,mode='M111')
    assert o['prediction'].shape==(1,12,388)
    assert o['candidate_weights'].shape==(1,13,12,388)
