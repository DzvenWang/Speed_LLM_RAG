"""v5.2.0 protocol and CRSI scientific invariants (not merely shapes)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pytest
import torch
from torch import nn

from tcrr_stla.evidence import counterfactual_residuals
from tcrr_stla.v5.calibration import chronological_partitions,choose_rejection_thresholds,assess_rejection_thresholds
from tcrr_stla.v5.regions import functional_region_ids,region_mean
from tcrr_stla.v5.diagnostics import select_matched_wrong_mask,MechanismDiagnostics
from tcrr_stla.v5.shared_adapter import s0_identity
from tcrr_stla.v5.losses import compute_losses
from test_v5 import model,data


class InputMaskTeacher(nn.Module):
    def forward(self,x,mask,tod,dow):
        # Explicitly mask-sensitive; detect fictitious new observations.
        pred=(x.float()*mask.float()).mean(1) + mask.float().mean(1)
        return pred[:,None].expand(-1,12,-1),None


def _candidate():
    x=torch.tensor([[[[4.,99.],[6.,12.],[8.,10.]]]],dtype=torch.float32)
    valid=torch.tensor([[[[1.,0.],[1.,1.],[0.,1.]]]],dtype=torch.float32)
    return dict(x=x,source_valid=valid,y=torch.zeros(1,1,12,2),
                tod=torch.zeros(1,1,3,dtype=torch.long),
                dow=torch.zeros(1,1,3,dtype=torch.long))


def test_query_all_observed_does_not_create_fictitious_counterfactual_response():
    c=_candidate();teacher=InputMaskTeacher();all_visible=torch.ones(1,3,2)
    normal=counterfactual_residuals(teacher,c,all_visible,replay='full',strict_source_mask=True)
    cf=counterfactual_residuals(teacher,c,all_visible,replay='matched',strict_source_mask=True)
    assert torch.equal(normal,cf)
    wrong=counterfactual_residuals(teacher,c,all_visible,replay='matched',strict_source_mask=False)
    assert (wrong-normal).abs().max()>1


def test_wrong_mask_comparison_only_changes_query_mask():
    c=_candidate();teacher=InputMaskTeacher()
    a=torch.tensor([[[1,0],[1,1],[0,1]]],dtype=torch.float32)
    b=torch.tensor([[[0,1],[1,0],[1,1]]],dtype=torch.float32)
    x=counterfactual_residuals(teacher,c,a,replay='matched',strict_source_mask=True)
    again=counterfactual_residuals(teacher,c,a.clone(),replay='matched',strict_source_mask=True)
    changed=counterfactual_residuals(teacher,c,b,replay='matched',strict_source_mask=True)
    assert torch.equal(x,again)
    assert not torch.equal(x,changed)


def test_wrong_mask_never_marks_identical_as_success():
    a=torch.ones(4,3,5)
    wrong,valid=select_matched_wrong_mask(a,['A']*4)
    assert not valid.any()
    assert torch.equal(a,wrong)


def test_wrong_mask_uses_real_pool_different_source_query():
    query=torch.ones(2,3,4)
    query[1,:1]=0
    pool=torch.ones(3,3,4)
    pool[0,:,2]=0
    pool[1,:2,1]=0
    pool[2,:,:2]=0
    wrong,changed=select_matched_wrong_mask(query,['A','B'],pool,['A','B','C'])
    assert changed.all()
    assert (query!=wrong).flatten(1).any(1).all()
    assert torch.equal(wrong[0],pool[0])
    assert torch.equal(wrong[1],pool[1])


def test_chronological_separation_purges_neighboring_windows():
    start=np.repeat(np.arange(300)*12,2)
    split=chronological_partitions(start,12,12)
    assert all(split[x] for x in ('selection','calibration','check'))
    a,b,c=[start[split[x]] for x in ('selection','calibration','check')]
    assert a.max()+24<b.min() and b.max()+24<c.min()
    assert len(set(split['selection']) & set(split['calibration']))==0
    assert len(set(split['calibration']) & set(split['check']))==0


def test_calibration_decision_does_not_use_locked_check_labels():
    n=300
    base=torch.zeros(n);proposal=torch.where(torch.arange(n)%2==0,torch.ones(n),-torch.ones(n))
    net=proposal.clone();risk=(proposal<0).float()
    fit=choose_rejection_thresholds(base,proposal,torch.ones(n),net,risk,
                                    torch.ones(n),torch.ones(n,dtype=torch.bool),min_coverage=.45)
    assert fit['calibration_mae_after']<fit['calibration_mae_before']
    assert fit['check_used'] is False
    check_a=assess_rejection_thresholds(base,proposal,torch.ones(n),net,risk,
         torch.ones(n),torch.ones(n,dtype=torch.bool),fit)
    check_b=assess_rejection_thresholds(base,proposal,torch.ones(n),net,risk,
         -torch.ones(n),torch.ones(n,dtype=torch.bool),fit)
    assert check_a!=check_b
    assert fit['enabled'] and fit['min_utility']==fit['min_utility']


def test_region_mean_is_identical_for_every_member():
    links=torch.tensor([[1,2],[0,2],[1,0],[4,5],[3,5],[4,3]])
    region=functional_region_ids(links,6,max_group_size=3)
    x=torch.randn(2,12,6,5)
    pooled=region_mean(x,region)
    for i in range(6):
        for j in range(6):
            if region[i]==region[j]:
                assert torch.equal(pooled[:,:,i],pooled[:,:,j])


def test_c3_coarse_logits_are_shared_over_region_horizon_blocks():
    m=model(n=6,p=8);m.eval()
    links=torch.tensor([[1,2],[0,2],[1,0],[4,5],[3,5],[4,3]])
    m.neighbours=links
    with torch.no_grad():
        m.c3.coarse_net[-1].weight.fill_(.025)
        m.c3.coarse_net[-1].bias.copy_(torch.tensor([.1,-.1]))
        out=m(**data(n=6,p=8,k=3),mode='M111',controller_variant='localized')
    coarse=out['coarse_intervention_logits']
    region=out['functional_region_ids']
    assert region is not None
    for h in (0,4):
        for i in range(6):
            for j in range(6):
                if region[i]==region[j]:
                    assert torch.allclose(coarse[:,h,i],coarse[:,h,j],atol=1e-7)
            for hh in range(h,h+4):
                assert torch.allclose(coarse[:,h],coarse[:,hh],atol=1e-7)


def test_joint_harm_is_supervised_with_actual_joint_control_result():
    m=model();m.set_stage('controller')
    o=m(**data(),mode='M111')
    b=dict(y_norm=torch.randn_like(o['prediction']),
           future_valid=torch.ones_like(o['prediction'],dtype=torch.bool))
    b['y_raw']=b['y_norm'].clone()
    loss,stats=compute_losses(o,b,0.,1.,{'training':{},'v5':{'joint_harm_weight':.2}},stage='controller')
    loss.backward()
    assert torch.isfinite(loss)
    assert 'joint_harm_bce' in stats
    assert m.c3.joint_harm[-1].weight.grad is not None
    assert m.c3.joint_harm[-1].weight.grad.abs().sum()>0


def test_new_c3_off_is_exact_noop():
    m=model();m.eval()
    a=m(**data(),mode='M110')
    m.c3.set_calibration(min_utility=100.,max_harm_probability=0.,enabled=True)
    b=m(**data(),mode='M110')
    assert torch.equal(a['prediction'],b['prediction'])
    assert torch.equal(a['prediction'],a['base_prediction']+a['proposed_correction'])


def test_s0_identity_changes_with_learning_settings():
    cfg={'v5':{'s0_epochs':8,'s0_lr':.0003},'training':{'batch_size':32}}
    a=s0_identity(cfg)
    cfg['v5']['s0_epochs']=9
    assert s0_identity(cfg)!=a
    cfg['v5']['s0_epochs']=8;cfg['v5']['s0_lr']=.0004
    assert s0_identity(cfg)!=a


def test_replay_manifest_checks_source_mask_protocol():
    source=Path(__file__).resolve().parents[1]/'tcrr_stla/replay_cache.py'
    text=source.read_text()
    assert "'replay_protocol':'source_mask_intersection_v2'" in text
    assert 'source_observed_sha256' in text and 'source_valid_sha256' in text
    assert "startswith(('4.','5.'))" in text


def test_risk_coverage_is_streamed_without_predictions_npz():
    o=model()(**data(),mode='M111')
    target=torch.randn_like(o['prediction'])
    agg=MechanismDiagnostics()
    agg.update(o,dict(y_norm=target,future_valid=torch.ones_like(target,dtype=torch.bool)))
    rows=agg.risk_coverage_rows()
    assert len(rows)==11 and rows[0]['correction_coverage']<=rows[-1]['correction_coverage']
    assert 'joint_harm_approx_auc' in agg.summary()
