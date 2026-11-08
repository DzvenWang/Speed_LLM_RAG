"""P0 diagnostics regression tests; no shared Backbone/data/GPU required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from tcrr_stla.p0 import query_metrics,candidate_probe_batch,aggregate_calibration,support_stratum


def test_query_paired_gain_and_mask():
    y=np.array([[[2.,4.],[2.,4.]],[[1.,3.],[1.,3.]]],np.float32)
    base=y+2.;full=y+1.;cross=y+3.;disabled=y+2.
    valid=np.ones_like(y,dtype=bool)
    valid[0,0,1]=False
    direct=np.array([[True,False],[False,True]],bool)
    output=query_metrics(y,valid,full,base,cross,disabled,None,direct)
    assert len(output)==2
    np.testing.assert_allclose(output['gain_mae'],[1,1])
    np.testing.assert_allclose(output['cross_history_gain'],[2,2])
    assert output['direct_masked_count'].tolist()==[2,2]
    assert output['not_direct_masked_count'].tolist()==[1,2]
    assert 'independent_gain' not in output


def test_query_nonfinite_predictions_common_support():
    y=np.array([[[0.,2.]]],np.float32)
    valid=np.ones_like(y,bool)
    full=np.array([[[np.nan,1.]]],np.float32)
    base=np.array([[[2.,2.]]],np.float32)
    row=query_metrics(y,valid,full,base,base,base)
    assert row.valid_count.iloc[0]==1
    assert row.full_mae.iloc[0]==1.
    assert row.base_mae.iloc[0]==0.


def test_oracle_candidate_and_calibration():
    # One candidate exactly removes +1 forecast error, another hurts it.
    device=torch.device('cpu')
    b={'y_norm':torch.tensor([[[2.,2.]]],device=device),
       'future_valid':torch.tensor([[[True,True]]],device=device)}
    base=torch.tensor([[[3.,3.]]])
    residual=torch.tensor([[[[-1.,1.]],[[1.,1.]]]])
    weights=torch.tensor([[[[.8,.8]],[[.1,.1]],[[.1,.1]]]])
    o={'candidate_residuals':residual,'base_prediction':base,
       'prediction':base+torch.tensor([[[-.5,.5]]]),
       'candidate_available':torch.ones_like(residual,dtype=torch.bool),
       'candidate_weights':weights,
       'risk_probability':torch.tensor([[[.7,.8]]]),
       'ungated_correction':torch.tensor([[[-1.,1.]]]),
       'gate':torch.tensor([[[.9,.9]]]),
       'router':None}
    values,bins=candidate_probe_batch(o,b,max_delta=100.,margin=.02)
    assert values['best_single_candidate_gain_norm'][0] > .3
    assert values['full_beneficial_element_rate'][0] == .5
    assert values['full_harmful_element_rate'][0] == .5
    assert sum(row['count'] for row in bins)==2
    tab=aggregate_calibration([{'severity':20,**v} for v in bins])
    assert int(tab['count'].sum())==2


def test_oracle_no_candidate_and_none_weight():
    b={'y_norm':torch.zeros((1,1,2)), 'future_valid':torch.ones((1,1,2),dtype=torch.bool)}
    o={'candidate_residuals':torch.ones((1,1,1,2)),
       'candidate_available':torch.zeros((1,1,1,2),dtype=torch.bool),
       'candidate_weights':torch.tensor([[[[0.,0.]],[[1.,1.]]]]),
       'base_prediction':torch.ones((1,1,2)), 'prediction':torch.ones((1,1,2)),
       'risk_probability':torch.zeros((1,1,2)), 'ungated_correction':torch.zeros((1,1,2)),
       'gate':torch.zeros((1,1,2)),'router':None}
    row,_=candidate_probe_batch(o,b,max_delta=3)
    assert row['best_single_candidate_gain_norm'][0]==0.
    assert row['candidate_available_rate'][0]==0.
    assert row['none_weight_mean'][0]==1.


def test_support_bins():
    assert support_stratum(.08)=='coverage_<10%'
    assert support_stratum(.6)=='coverage_50-70%'


def test_checkpoint_protocol_validation_and_fingerprint(tmp_path):
    from tcrr_stla.p0 import validate_pair,read_predictions,sha256
    c={'experiment':dict(target='speed',dataset='TraffiDent_Net_06',seed=10,history=12,horizon=12,stride=12,backbone_mode='incomplete'),
       'training':dict(ablation='full',batch_size=64,warmup_epochs=10,partial_epochs=35,max_epochs=100,
                        patience=10,lr_backbone=1e-5,lr_new=1e-3,weight_decay=.01,grad_clip=1.)}
    import copy,json
    first={'version':'2.0.0','config':c,'backbone_sha256':'same-sha'}
    control=copy.deepcopy(first);control['config']['training']['ablation']='no_retrieval'
    a=tmp_path/'full'/'best.pt';b=tmp_path/'none'/'best.pt';a.parent.mkdir();b.parent.mkdir()
    torch.save(first,a);torch.save(control,b)
    _,v=validate_pair(a,b,'TCRR-STLA')
    assert v['config']['training']['ablation']=='no_retrieval'
    path=a.parent/'evaluation'/'s20'/'full';path.mkdir(parents=True)
    np.savez_compressed(path/'test_predictions.npz',prediction=np.zeros((2,1,1)),base_prediction=np.zeros((2,1,1)))
    (path/'summary.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(a))))
    assert read_predictions(a,20,'full',sha256(a))[0].shape==(2,1,1)
    with __import__('pytest').raises(RuntimeError):read_predictions(a,20,'full','incorrect')
    control['backbone_sha256']='different';torch.save(control,b)
    with __import__('pytest').raises(ValueError):validate_pair(a,b,'TCRR-STLA')
