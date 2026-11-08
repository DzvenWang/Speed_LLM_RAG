"""Three orthogonal mechanism switches, tested on the same weights and evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from test_v5 import model,data
from tcrr_stla.v5.model import MODES,THREE_SWITCH_MODES,ALL_MODES,resolve_mechanism_mode
from tcrr_stla.v5.losses import compute_losses

@pytest.mark.parametrize('name,flags',sorted(THREE_SWITCH_MODES.items()))
def test_three_switch_controls_on_one_checkpoint(name,flags):
    m=model();m.set_stage('controller');m.eval();d=data()
    with torch.no_grad():
        o=m(**d,mode=name)
    cf,sup,c3=flags
    assert o['prediction'].shape==o['base_prediction'].shape
    assert o['c1_enabled']==cf
    assert o['c2_enabled']==sup
    assert o['c3_enabled']==c3
    assert o['controller_mode']==('learned' if c3 else 'none')
    if not cf:
        assert o['cf_increment'].count_nonzero()==0
    if not sup:
        assert o['support_logits_delta'].count_nonzero()==0
        assert o['support_compatibility_aggregated'].count_nonzero()==0
    if not c3:
        assert torch.equal(o['gate'],torch.ones_like(o['gate']))
        assert torch.equal(o['prediction'],o['base_prediction']+o['ungated_correction'])
    else:
        assert (o['gate']>=0).all() and (o['gate']<=1).all()
    assert torch.isfinite(o['prediction']).all()

@pytest.mark.parametrize('cf,sup',[(0,0),(0,1),(1,0),(1,1)])
def test_c3_off_independent_of_controller_parameters_and_calibration(cf,sup):
    m=model();m.eval();d=data()
    mode=f'M{cf}{sup}0'
    with torch.no_grad():
        before=m(**d,mode=mode)['prediction'].clone()
        # The controller's calibration is aggressive, but it must not apply
        # while the C3 branch is disabled.
        m.c3.set_calibration(min_utility=1000.,max_harm_probability=0.,enabled=True)
        for p in m.c3.parameters():p.add_(3.0*torch.randn_like(p))
        after=m(**d,mode=mode)['prediction']
    assert torch.allclose(before,after,atol=1e-7)

@pytest.mark.parametrize('cf,sup',[(0,0),(0,1),(1,0),(1,1)])
def test_c3_on_can_reject_to_original_backbone(cf,sup):
    m=model();d=data();m.eval()
    m.c3.set_calibration(min_utility=10000.,enabled=True)
    with torch.no_grad():
        o=m(**d,mode=f'M{cf}{sup}1')
    assert o['reject_correction'].all()
    assert torch.equal(o['prediction'],o['base_prediction'])


def test_three_switch_mode_rejects_conflicting_override():
    m=model();d=data()
    with pytest.raises(ValueError,match='requires controller_mode=learned'):
        m(**d,mode='M111',controller_mode='fixed')
    with pytest.raises(ValueError,match='requires controller_mode=none'):
        m(**d,mode='M110',controller_mode='learned')
    with pytest.raises(ValueError,match='Expected one of'):
        m(**d,mode='M120')


def test_legacy_four_modes_remain_fixed_gate_in_c12_stage():
    m=model();d=data()
    for legacy_mode in MODES:
        out=m(**d,mode=legacy_mode)
        assert torch.allclose(out['gate'],torch.full_like(out['gate'],.75))


def test_eight_modes_backprop_without_c3_leakage_in_off_mode():
    m=model();m.set_stage('joint');d=data()
    for mode in THREE_SWITCH_MODES:
        m.zero_grad(set_to_none=True)
        out=m(**d,mode=mode)
        pred=out['prediction']
        loss=pred.square().sum()
        loss.backward()
        any_c3_grad=any(p.grad is not None and p.grad.abs().sum().item()>0 for p in m.c3.parameters())
        assert any_c3_grad == (mode[-1]=='1')


def test_all_eight_are_declared_and_unique():
    assert len(THREE_SWITCH_MODES)==8
    assert len(ALL_MODES)==12
    assert set(THREE_SWITCH_MODES)=={f'M{i}{j}{k}' for i in (0,1) for j in (0,1) for k in (0,1)}
    assert resolve_mechanism_mode('M111','c12',None)==(True,True,True,'learned')
    assert resolve_mechanism_mode('M110','controller',None)==(True,True,False,'none')


def test_controller_validation_emits_eight_switch_maes(monkeypatch):
    from tcrr_stla.v5 import trainer
    m=model();m.set_stage('controller');m.eval();d=data()
    B,H,N=d['x'].shape
    y=torch.randn(B,12,N)
    batch={k:d[k] for k in ('x','mask','tod','dow')}
    batch.update(index=torch.arange(B),y_raw=y,y_norm=y,
                 future_valid=torch.ones_like(y,dtype=torch.bool))
    ev={k:d[k] for k in ('candidate_y','residual','candidate_scores',
        'candidate_coverage','candidate_valid','normal_residual',
        'candidate_histories','candidate_source_valid')}
    monkeypatch.setattr(trainer,'_batch_to_device',lambda b,device:b)
    monkeypatch.setattr(trainer,'evidence_for_batch',lambda *args,**kw:ev)

    class FakeBaseCache:
        def fetch(self,index,device):
            return torch.zeros(B,12,N),torch.zeros(B,N,24)

    config={'v5':{},'training':{},'runtime':{}}
    scores=trainer.run_epoch(m,[batch],None,None,torch.device('cpu'),0.,1.,config,
                             'controller',FakeBaseCache(),None,None,None,epoch=1)
    for mode in THREE_SWITCH_MODES:
        assert f'{mode}_mae' in scores
    assert scores['M111_mae']==pytest.approx(scores['mae'])
    for cf in (0,1):
        for sup in (0,1):
            expected=scores[f'M{cf}{sup}0_mae']-scores[f'M{cf}{sup}1_mae']
            assert scores[f'c3_gain_cf{cf}_support{sup}']==pytest.approx(expected)
