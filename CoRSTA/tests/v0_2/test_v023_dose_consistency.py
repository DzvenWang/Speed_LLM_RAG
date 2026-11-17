"""CoRSTA v0.2.3: Counterfactual Dose Ladder (B) and view-consistency reliability (A).

Synthetic, CPU-only (FakeSTLA).  Run:  python -m pytest tests/v0_2/test_v023_dose_consistency.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pytest
import torch
import yaml
from torch.utils.data import Dataset
from test_v2_core import FakeSTLA,graph,batch
from corsta_v2.model import CoRSTAv2
from corsta_v2.mechanisms import mechanism_kwargs
from corsta_v2.counterfactual.dose import (CONSISTENCY_DIM,validate_levels,nested_restore_masks,
                                           dose_psi,consistency_features,select_dose)
from corsta_v2.counterfactual.teacher import PairedTeacher
from corsta_v2.training.normal_anchor import normal_anchor_loss
from corsta_v2.training.evaluation import evaluate

torch.set_num_threads(1)
LEVELS=(0.25,0.5,0.75,1.0)
CONFIG={'training':{'forecast_rmse_weight':0.05,'normal_residual_weight':0.1,
                    'cf_response_weight':0.06,'cf_error_gain_weight':0.03,
                    'cf_anchor_gain_weight':0.05,'cf_acceptance_weight':0.02,
                    'cf_error_temperature':0.35,'cf_error_rank_weight':0.02,
                    'cf_anchor_rank_weight':0.02,'cf_latent_weight':0.01,
                    'dose_response_weight':0.04,'dose_gain_weight':0.05,'dose_gain_scale':0.1}}


def make(levels=LEVELS,views=2,stage='anchor_cf',seed=5):
    torch.manual_seed(seed)
    m=CoRSTAv2(FakeSTLA(),graph(),hidden=16,failure_dim=8,cf_dim=8,residual_dim=8,horizon=4,heads=2,
               prediction_mode='normal_anchored',dose_levels=levels,dose_temperature=0.01,
               consistency_views=views,consistency_drop=0.3,consistency_seed=3)
    m.set_stage(stage,use_graph=False)
    return m


# ---------------------------------------------------------------- pure helpers
def test_validate_levels():
    assert validate_levels([0.5,1.0])==(0.5,1.0)
    for bad in ([],[0.0,1.0],[0.5],[0.5,0.5,1.0],[1.0,0.5],[0.5,1.2]):
        with pytest.raises(ValueError):
            validate_levels(bad)


def test_nested_restore_masks_are_nested_and_bounded():
    torch.manual_seed(0)
    original=torch.ones(3,12,40)
    mask=(torch.rand(3,12,40)>0.5).float()
    original[0,0,:5]=0;mask[0,0,:5]=0   # never observed: must never be restored
    rungs=nested_restore_masks(mask,original,(0.25,0.5,0.75))
    assert rungs.shape==(3,3,12,40)
    for j in range(3):
        assert torch.all(rungs[:,j]>=mask)
        assert torch.all(rungs[:,j]<=original)
    assert torch.all(rungs[:,0]<=rungs[:,1]) and torch.all(rungs[:,1]<=rungs[:,2])
    missing=((original>0.5)&(mask<0.5)).sum().item()
    for j,d in enumerate((0.25,0.5,0.75)):
        restored=(rungs[:,j]-mask).sum().item()
        assert abs(restored/missing-d)<0.08
    full=nested_restore_masks(mask,original,(1.0,))[:,0]
    assert torch.equal(full,original)


def test_dose_psi_monotone_and_exact_at_full_dose():
    gamma=torch.rand(2,4,5)*3.8+0.25
    psi=dose_psi(torch.tensor(LEVELS),gamma)
    assert psi.shape==(2,4,4,5)
    assert torch.all(psi[:,1:]>=psi[:,:-1])
    assert torch.equal(psi[:,-1],torch.ones(2,4,5))
    assert torch.all((psi>0)&(psi<=1))
    linear=dose_psi(torch.tensor(LEVELS),torch.ones(2,4,5))
    assert torch.allclose(linear[:,0],torch.full((2,4,5),0.25))


def test_consistency_features_semantics():
    base0=torch.randn(2,4,5);delta0=torch.randn(2,4,5)
    # identical views: no disagreement anywhere
    same=consistency_features(base0,delta0,base0[None].repeat(3,1,1,1),delta0[None].repeat(3,1,1,1))
    assert same.shape==(2,4,5,CONSISTENCY_DIM)
    assert torch.allclose(same[...,0],torch.zeros(2,4,5)) and torch.allclose(same[...,1],torch.zeros(2,4,5))
    assert torch.allclose(same[...,2],torch.zeros(2,4,5)) and torch.allclose(same[...,3],torch.ones(2,4,5))
    # a correction that exactly compensates each view's drift contracts disagreement fully
    drift=0.5+torch.rand(3,2,4,5)   # bounded away from 0 so ratios are well conditioned
    contracted=consistency_features(base0,delta0,base0[None]+drift,delta0[None]-drift)
    assert torch.all(contracted[...,2]>0.99)
    # a correction that amplifies drift is flagged unreliable
    amplified=consistency_features(base0,delta0,base0[None]+drift,delta0[None]+drift)
    assert torch.all(amplified[...,2]<0)
    # sign agreement
    flipped=consistency_features(base0,delta0,base0[None].repeat(2,1,1,1),(-delta0)[None].repeat(2,1,1,1))
    assert torch.all(flipped[...,3]<1e-6)
    assert not same.requires_grad


def test_select_dose_hard_and_soft():
    psi=dose_psi(torch.tensor(LEVELS),torch.ones(1,2,3))
    gains=torch.zeros(1,4,2,3)
    gains[0,2]=0.05;gains[0,1,0,0]=0.2;gains[0,:,1,2]=-0.1
    sel,lvl=select_dose(gains,psi,torch.tensor(LEVELS),margin=0.0)
    assert sel[0,0,0].item()==pytest.approx(0.5) and lvl[0,0,0].item()==pytest.approx(0.5)
    assert sel[0,0,1].item()==pytest.approx(0.75)
    assert sel[0,1,2].item()==0.0 and lvl[0,1,2].item()==0.0      # all negative: reject
    rejected,_=select_dose(torch.zeros(1,4,2,3),psi,torch.tensor(LEVELS),margin=0.0)
    assert torch.equal(rejected,torch.zeros(1,2,3))                 # zero gain never exceeds margin
    soft,_=select_dose(gains,psi,torch.tensor(LEVELS),margin=0.0,temperature=1e-4)
    assert torch.allclose(soft,sel,atol=1e-3)
    gains.requires_grad_(True)
    soft,_=select_dose(gains,psi,torch.tensor(LEVELS),margin=0.0,temperature=0.05)
    soft.sum().backward()
    assert gains.grad.abs().sum()>0


# --------------------------------------------------------------------- teacher
def test_teacher_dose_rungs_between_masked_and_full():
    m=make(stage='anchor_cf')
    obs,b=batch(B=3)
    out=m(obs,enable_controller=False)
    teacher=PairedTeacher(m.stla)
    base=teacher(b,out['h0'],out['backbone'])
    assert 'dose_p' not in base
    teacher.dose_levels=LEVELS[:-1]
    t=teacher(b,out['h0'],out['backbone'])
    assert t['dose_p'].shape==(3,3,4,5)
    assert torch.isfinite(t['dose_p']).all()
    teacher.dose_levels=(1.0,)
    full=teacher(b,out['h0'],out['backbone'])['dose_p'][:,0]
    assert torch.allclose(full,t['p_full'],atol=1e-5)               # d=1 rung IS P_full
    b2=dict(b);b2['original_valid']=b['mask'].clone()               # nothing restorable
    teacher.dose_levels=(0.5,)
    none=teacher(b2,out['h0'],out['backbone'])['dose_p'][:,0]
    assert torch.allclose(none,out['backbone'].detach(),atol=1e-5)  # d>0 with nothing missing == P0


# ----------------------------------------------------------------------- model
def test_disabled_mechanisms_keep_v022_outputs():
    torch.manual_seed(5)
    plain=CoRSTAv2(FakeSTLA(),graph(),hidden=16,failure_dim=8,cf_dim=8,residual_dim=8,horizon=4,heads=2,
                   prediction_mode='normal_anchored')
    obs,_=batch()
    out=plain(obs)
    assert out['dose_psi'] is None and out['dose_gain_pred'] is None and out['consistency'] is None
    assert not plain.dose_enabled
    assert not any('dose' in n for n,_ in plain.named_parameters())


def test_cf_first_rejects_new_mechanisms():
    with pytest.raises(ValueError):
        CoRSTAv2(FakeSTLA(),graph(),hidden=16,failure_dim=8,cf_dim=8,residual_dim=8,horizon=4,heads=2,
                 prediction_mode='cf_first',dose_levels=LEVELS)


def test_forward_shapes_and_identity_at_init():
    m=make(stage='anchor_control');m.eval()
    obs,_=batch()
    with torch.no_grad():
        out=m(obs,enable_controller=True)
        assert out['dose_psi'].shape==(2,4,4,5)
        assert out['dose_gain_pred'].shape==(2,4,4,5)
        assert out['consistency'].shape==(2,4,5,CONSISTENCY_DIM)
        assert not out['consistency'].requires_grad
        # zero-initialised policy: predicted gain 0 never exceeds margin 0 -> falls back to Normal
        assert torch.allclose(out['prediction'],out['normal_prediction'],atol=1e-6)
        assert torch.equal(out['dose_selected_level'],torch.zeros(2,4,5))
        # zero-initialised gamma: psi_d == d
        assert torch.allclose(out['dose_psi'][:,0],torch.full((2,4,5),0.25))
        # CF-training behaviour is unchanged: no controller => full-dose candidate
        free=m(obs,enable_controller=False)
        assert torch.allclose(free['prediction'],free['normal_cf_candidate'],atol=1e-6)
        assert free['dose_selected_level'] is None


def test_policy_selects_first_rung_when_all_gains_tie_positive():
    m=make(stage='anchor_control');m.eval()
    obs,_=batch()
    with torch.no_grad():
        m.cf.dose_policy[-1].bias.fill_(1.0)
        out=m(obs,enable_controller=True)
        expected=out['normal_prediction']+0.25*out['delta_cf']
        assert torch.allclose(out['prediction'],expected,atol=1e-6)
        assert torch.allclose(out['dose_selected_level'],torch.full((2,4,5),0.25))


def test_consistency_is_deterministic_in_eval_and_label_free():
    m=make(stage='anchor_control');m.eval()
    obs,_=batch()
    with torch.no_grad():
        a=m(obs)['consistency'];b=m(obs)['consistency']
        assert torch.equal(a,b)
        assert torch.isfinite(a).all()
        # the forward only accepts observed inputs, so no label/full-history path exists
        with pytest.raises(ValueError):
            m({**obs,'y_norm':torch.zeros(2,4,5)})


def test_view_consistency_does_not_change_primary_path():
    on=make(views=2,stage='anchor_cf');off=make(views=0,stage='anchor_cf')
    on.eval();off.eval()
    off.load_state_dict({k:v for k,v in on.state_dict().items() if k in off.state_dict()
                         and v.shape==off.state_dict()[k].shape},strict=False)
    obs,_=batch()
    with torch.no_grad():
        a=on(obs,enable_controller=False);b=off(obs,enable_controller=False)
    assert torch.allclose(a['normal_prediction'],b['normal_prediction'],atol=1e-6)
    assert torch.allclose(a['delta_cf'],b['delta_cf'],atol=1e-6)


def test_stage_trainability_of_new_heads():
    cf=make(stage='anchor_cf')
    names=[n for n,p in cf.named_parameters() if p.requires_grad]
    assert any('dose_head' in n for n in names) and any('dose_policy' in n for n in names)
    assert not any(n.startswith(('failure','residual','graph_residual','stla')) for n in names)
    ctrl=make(stage='anchor_control')
    names=[n for n,p in ctrl.named_parameters() if p.requires_grad]
    assert names and all(('anchor_gain_head' in n or 'anchor_acceptance_head' in n or 'dose_policy' in n)
                         for n in names)
    assert not any('dose_head' in n for n in names)                 # response curve frozen in control


# ---------------------------------------------------------------------- losses
def test_anchor_cf_loss_has_dose_terms_and_gradients():
    m=make(stage='anchor_cf');m.train()
    obs,b=batch()
    out=m(obs,enable_controller=False)
    teacher=PairedTeacher(m.stla);teacher.dose_levels=LEVELS[:-1]
    t=teacher(b,out['h0'],out['backbone'])
    other={k:v.clone() for k,v in obs.items()};other['mask'][:,0]=0;other['x'][:,0]=0
    alt=m(other,enable_controller=False)
    loss,d=normal_anchor_loss(out,b,t,'anchor_cf',0,1,CONFIG,other_views=[alt])
    assert torch.isfinite(loss)
    assert 'dose_response' in d and 'dose_gain' in d
    loss.backward()
    assert m.cf.dose_head.weight.grad is not None and m.cf.dose_head.weight.grad.abs().sum()>0
    assert m.cf.dose_policy[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.stla.parameters())
    assert all(p.grad is None for p in m.residual.parameters())


def test_anchor_control_loss_trains_only_policy_and_anchor_heads():
    m=make(stage='anchor_control');m.train()
    obs,b=batch()
    out=m(obs,enable_controller=True)
    teacher=PairedTeacher(m.stla)
    t=teacher(b,out['h0'],out['backbone'])
    loss,d=normal_anchor_loss(out,b,t,'anchor_control',0,1,CONFIG)
    assert torch.isfinite(loss) and 'dose_gain' in d and 'dose_applied_level' in d
    assert 'dose_response' not in d
    loss.backward()
    assert m.cf.dose_policy[-1].weight.grad.abs().sum()>0
    assert m.cf.dose_head.weight.grad is None
    assert m.cf.shift[-1].weight.grad is None


def test_dose_response_loss_zero_when_prediction_matches_teacher():
    m=make(stage='anchor_cf')
    obs,b=batch()
    out=m(obs,enable_controller=False)
    from corsta_v2.training.normal_anchor import dose_response_loss
    n=len(LEVELS)-1
    perfect=(out['dose_psi'][:,:n]*out['delta_cf_candidate'][:,None]+out['backbone'][:,None]).detach()
    value=dose_response_loss(out,perfect,torch.ones(2,4,5))
    assert value.item()<1e-9


# ------------------------------------------------------------------ evaluation
class _Tiny(Dataset):
    def __init__(self,n=4):
        self.items=[]
        for _ in range(n):
            obs,b=batch(B=1)
            self.items.append({k:v[0] for k,v in b.items() if k in
                               ('x','mask','tod','dow','y_norm','y_raw','future_valid')})
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]


def test_evaluate_reports_dose_and_consistency_diagnostics():
    m=make(stage='anchor_control');m.eval()
    result=evaluate(m,{'40':_Tiny()},torch.device('cpu'),0.0,1.0,batch_size=2,mode='anchor_control')
    diag=result['per_severity']['40']['cf_diagnostics']
    assert 'dose' in diag and set(diag['dose']['level_fraction'])=={str(v) for v in LEVELS}
    assert diag['dose']['reject_fraction']==pytest.approx(1.0)       # zero-init policy rejects
    assert len(diag['consistency_bins'])==4
    assert sum(b['fraction'] for b in diag['consistency_bins'])==pytest.approx(1.0)
    import json
    json.dumps(result,allow_nan=False)


# ---------------------------------------------------------------------- config
@pytest.mark.parametrize('name,dose,views',[('corsta_v023',4,3),('corsta_v023_B_only',4,0),
                                            ('corsta_v023_A_only',1,3),('corsta_v023_off',0,0)])
def test_shipped_configs_map_to_expected_mechanisms(name,dose,views):
    cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/v0_2'/f'{name}.yaml').read_text())
    kw=mechanism_kwargs(cfg)
    assert len(kw.get('dose_levels',()))==dose
    assert kw.get('consistency_views',0)==views


def test_view_consistency_requires_dose_ladder():
    cfg={'model':{'view_consistency':{'enabled':True},'dose_ladder':{'enabled':False}},'experiment':{}}
    with pytest.raises(RuntimeError):
        mechanism_kwargs(cfg)
