"""Exact parity against the user-supplied Base_SupportBench public API."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from fare.evaluation import _metrics, common_metrics
from scripts.report_retrieval_attribution import build_report


def test_metric_bundle_is_the_only_metric_definition():
    bundle, per_query, proto, threshold = common_metrics()
    assert threshold == 1.0
    assert proto == 'raw_target_percent_exclude_abs_le_1_v1'
    assert callable(bundle) and callable(per_query)
    pred = np.full((3,12,3),10.0)
    true = np.full_like(pred,20.0)
    true[0,:,0]=0.0
    true[1,:,1]=0.5
    true[2,2,0]=np.nan
    pred[2,5,1]=np.nan
    valid = np.ones_like(pred,bool)
    valid[0,1,2]=False
    direct = np.array([[1,0,1],[0,1,0],[1,0,1]],bool)
    canonical=bundle(pred,true,valid,direct)
    actual=_metrics(pred,true,valid,direct)
    for key,value in canonical.items():
        assert key in actual, key
        if isinstance(value,(int,float,np.floating)) and not isinstance(value,bool):
            assert actual[key] == pytest.approx(value, nan_ok=True), key
        else:
            assert actual[key] == value, key
    for key in ('MAE','RMSE','MAPE','count','MAPE_count','MAPE_coverage'):
        assert actual[key] == pytest.approx(canonical['overall_'+key],nan_ok=True)
        assert actual['directly_masked_'+key] == pytest.approx(canonical['masked_'+key],nan_ok=True)
    for h in (3,6,12):
        for metric in ('MAE','RMSE','MAPE','MAPE_count','MAPE_coverage'):
            assert actual[f'H{h}_{metric}'] == pytest.approx(canonical[f'H{h}_overall_{metric}'],nan_ok=True)
            assert actual[f'H{h}_directly_masked_{metric}'] == pytest.approx(canonical[f'H{h}_masked_{metric}'],nan_ok=True)


def test_h3_is_step_three_not_three_step_average():
    y=np.ones((1,12,1))*10
    p=y.copy()
    p[:,2]=15
    a=_metrics(p,y,np.ones_like(y,bool),np.ones((1,1),bool))
    assert a['H3_MAPE']==pytest.approx(50.0)
    assert a['H6_MAPE']==pytest.approx(0.0)
    assert a['H12_MAPE']==pytest.approx(0.0)
    assert a['MAPE']==pytest.approx(50/12)


def test_empty_mape_is_nan_with_zero_count_and_coverage():
    pred=np.full((1,12,2),7.)
    target=np.zeros_like(pred)
    mask=np.ones_like(pred,bool)
    direct=np.ones((1,2),bool)
    result=_metrics(pred,target,mask,direct)
    assert math.isnan(result['MAPE'])
    assert result['MAPE_count']==0
    assert result['MAPE_coverage']==0.0
    assert result['MAE']==7.0
    mask[:]=False
    no_valid=_metrics(pred,target,mask,direct)
    assert math.isnan(no_valid['MAPE'])
    assert math.isnan(no_valid['MAPE_coverage'])
    assert no_valid['count']==0


def test_shape_checks_prevent_invalid_node_masks():
    pred=np.zeros((2,12,5))
    with pytest.raises(ValueError,match='direct_mask'):
        _metrics(pred,pred,np.ones_like(pred,bool),np.ones((2,12,5),bool))


def test_retrieval_report_keeps_mape_in_macro_column():
    def r(mode,severity,mae,mape):
        return dict(mode=mode,split='test',severity=severity,MAE=mae,RMSE=mae+1,
                    MAPE=mape,MAPE_count=20,MAPE_coverage=.8,
                    mape_protocol='raw_target_percent_exclude_abs_le_1_v1')
    rows=[r('full',20,2.,15.),r('similarity_memory',20,3.,30.),
          r('full',40,4.,45.),r('similarity_memory',40,5.,60.)]
    md=build_report(rows)
    assert '| full | 3.00000 | 4.00000 | 30.00 | +0.00% |' in md
    assert '| similarity_memory | 4.00000 | 5.00000 | 45.00 | +25.00% |' in md
    assert 'MAPE coverage' in md


def test_retrieval_report_rejects_wrong_protocol():
    with pytest.raises(ValueError,match='shared Base_SupportBench'):
        build_report([dict(mode='full',split='val',MAE=1.,RMSE=2.,MAPE=3.,mape_protocol='other')])
