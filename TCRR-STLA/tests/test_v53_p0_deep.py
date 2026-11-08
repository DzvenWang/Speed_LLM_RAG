"""Regression tests for read-only paired P0 v5.3 diagnostics."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tcrr_stla.v5.p0_deep_diagnostics import P0DeepDiagnostics
from scripts.p0_summarize_v5 import block_bootstrap,weighted_mean


def make_batch():
    b,p,n,k=2,3,4,2
    base=torch.zeros(b,p,n)
    y=torch.ones(b,p,n)
    valid=torch.ones(b,p,n,dtype=torch.bool)
    valid[1,2,3]=False
    def out(pred):return {'prediction':torch.full_like(base,pred)}
    m110=out(.6)
    m110.update(base_prediction=base,
                candidate_residuals=torch.stack([torch.ones_like(base)*.8,torch.ones_like(base)*(-.8)],dim=1),
                candidate_available=torch.ones(b,k,p,n,dtype=torch.bool),
                candidate_weights=torch.stack([torch.ones_like(base)*.7,torch.ones_like(base)*.2,
                                               torch.ones_like(base)*.1],dim=1),
                candidate_expected_net_gain=torch.stack([torch.ones_like(base)*.7,
                                                        -torch.ones_like(base)*.5],dim=1),
                initial_proposal=torch.ones_like(base)*.55,
                proposed_correction=torch.ones_like(base)*.6,
                residual_bound=3.)
    m111=out(.7)
    m111.update(base_prediction=base,pre_rejection_correction=torch.ones_like(base)*.7,
                harm_probability=torch.ones_like(base)*.2,
                normal_gate=torch.ones_like(base)*.7,
                cf_gate=torch.ones_like(base)*.4,
                reject_correction=torch.zeros_like(valid),
                component_conflict=torch.zeros_like(base))
    outputs={'M000':out(0), 'M100':out(.2), 'M110':m110,'M111':m111}
    batch=dict(y_raw=y,future_valid=valid,mask=torch.ones(b,12,n),index=torch.tensor([100,101]))
    return outputs,batch


def test_paired_gain_positive_and_candidate_headroom(tmp_path):
    outputs,batch=make_batch()
    audit=P0DeepDiagnostics(mean=0,std=1)
    audit.update(outputs,batch,families=['f1','f2'])
    report=audit.save(tmp_path/'results',dict(target='speed',split='validation'))
    assert report['queries']==2 and report['valid_positions']==23
    assert report['candidate_valid_positions']==46
    assert report['metrics']['c1_gain_M000_to_M100']==pytest.approx(.2,abs=1e-6)
    assert report['metrics']['c2_gain_M100_to_M110']==pytest.approx(.4,abs=1e-6)
    assert report['metrics']['c3_gain_M110_to_M111']==pytest.approx(.1,abs=1e-6)
    assert report['metrics']['c3_harm_rate']==0
    assert report['metrics']['candidate_positive_rate']==pytest.approx(.5)
    assert report['metrics']['support_none_weight']==pytest.approx(.1)
    assert report['metrics']['support_harmful_weight_share']==pytest.approx(.2/.9)
    assert report['harm_risk_vs_backbone']['positions']==23
    assert len(list(csv.DictReader((tmp_path/'results/per_query.csv').open())))==2
    grouped=list(csv.DictReader((tmp_path/'results/grouped_c3.csv').open()))
    assert {r['group_kind'] for r in grouped}>={'horizon','observation_ratio','fault_family','component_conflict','proposal_magnitude_norm'}
    assert (tmp_path/'results/worst_c3_queries.csv').exists()
    risk=list(csv.DictReader((tmp_path/'results/hypothetical_risk_coverage.csv').open()))
    assert len(risk)==21
    assert float(risk[-1]['coverage'])==1.0


def test_wrong_masks_and_duplicate_indices_rejected():
    outputs,batch=make_batch()
    audit=P0DeepDiagnostics(mean=0,std=1)
    audit.update(outputs,batch)
    with pytest.raises(ValueError,match='exactly once'):
        audit.update(outputs,batch)
    outputs,batch=make_batch()
    batch['future_valid']=torch.ones(2,2,4,dtype=torch.bool)
    with pytest.raises(ValueError,match='matching'):
        P0DeepDiagnostics(mean=0,std=1).update(outputs,batch)


def test_no_candidate_available_uses_none_oracle_and_no_nan():
    outputs,batch=make_batch()
    outputs['M110']['candidate_available'].fill_(False)
    outputs['M110']['candidate_weights'][:,0].fill_(0)
    outputs['M110']['candidate_weights'][:,1].fill_(0)
    outputs['M110']['candidate_weights'][:,2].fill_(1)
    audit=P0DeepDiagnostics(mean=0,std=1)
    audit.update(outputs,batch)
    s=audit.summary()
    assert s['metrics']['candidate_oracle_single_gain_over_backbone']==0
    assert s['metrics']['support_none_weight']==pytest.approx(1)
    assert s['candidate_expected_gain_pearson'] is None


def test_harm_attribution_and_harm_score(tmp_path):
    outputs,batch=make_batch()
    outputs['M111']['prediction'][:]=.0 # C3 worsens the M110 correction
    outputs['M111']['harm_probability'][:]=.9
    audit=P0DeepDiagnostics(mean=0,std=1)
    audit.update(outputs,batch)
    assert audit.summary()['metrics']['c3_gain_M110_to_M111']==pytest.approx(-.6)
    assert audit.summary()['metrics']['c3_harm_rate']==pytest.approx(1)
    assert audit.summary()['metrics']['c3_harms_previously_beneficial_M110']==pytest.approx(1)
    assert audit.summary()['harm_risk_vs_c3_off']['positive_rate']==pytest.approx(1)


def test_block_bootstrap_resamples_queries_not_pixels():
    rows=[dict(dataset_index=str(i),valid_positions='10',c3_gain=str(.1 if i<30 else -.2))
          for i in range(60)]
    assert weighted_mean(rows,'c3_gain')==pytest.approx(-.05)
    interval=block_bootstrap(rows,'c3_gain',seed=10,block_size=10,repeats=100)
    assert interval[0] <= -.05 <= interval[1]
    assert block_bootstrap(rows,'c3_gain',seed=10,block_size=10,repeats=100)==interval
    assert block_bootstrap(rows,'c3_gain',seed=10,block_size=100,repeats=100)==(None,None)


def test_p0_mode_contract_and_no_training_calls():
    evaluator=(ROOT/'scripts/evaluate_v5.py').read_text()
    pbs=(ROOT/'pbs/p0_deep_diagnostics_v5.pbs').read_text()
    for symbol in ('--p0-deep','P0DeepDiagnostics','p0.update(outputs,b','p0.save('):
        assert symbol in evaluator
    for symbol in ('--split validation','--split test','--validation-section',
                   '--modes M000 M100 M110 M111','--p0-deep','best_controller.pt'):
        assert symbol in pbs
    assert 'train_v5.py' not in pbs
    assert 'optimizer.step(' not in evaluator
    assert 'optimizer.step(' not in (ROOT/'tcrr_stla/v5/p0_deep_diagnostics.py').read_text()
