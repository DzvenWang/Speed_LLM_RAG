"""Regression for v1.1's false (1,6) assertion on an actual seven-feature vector."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostic.gain import (ForecastArrays, LOCAL_FEATURE_DIM,
                             LOCAL_FEATURE_NAMES, edge_horizon_features,
                             audit_forecast_arrays)


def make_arr(R=1, H=12, P=12, N=388):
    rng=np.random.default_rng(42)
    a=ForecastArrays(
        starts=np.arange(R,dtype=np.int64),
        pred=np.zeros((R,P,N),dtype=np.float32),
        truth=np.ones((R,P,N),dtype=np.float32),
        future_valid=np.ones((R,P,N),dtype=bool),
        x=rng.normal(size=(R,H,N)).astype(np.float32),
        observed=np.ones((R,H,N),dtype=bool),
    )
    return a


def test_real_preflight_contract_uses_seven_local_features():
    assert LOCAL_FEATURE_DIM == 7
    assert len(set(LOCAL_FEATURE_NAMES)) == 7
    for split in ('TRAIN', 'VAL'):
        a=make_arr()
        c=audit_forecast_arrays(a,split=split,max_lag=6)
        assert c['local_feature_dim']==7 and c['checked_pairs']>=4
        features, src, eligible=edge_horizon_features(a,0,1,1,1)
        assert features.shape == (1,7) and src.shape==(1,) and eligible.shape==(1,)


def test_source_mask_and_future_validity_affect_eligibility_not_shape():
    a=make_arr(R=3)
    a.observed[0,-1,0]=False
    a.future_valid[1,0,1]=False
    c=audit_forecast_arrays(a,split='TRAIN',max_lag=6)
    assert c['rows']==3
    X,s,elig=edge_horizon_features(a,0,1,1,1)
    assert X.shape == (3,7)
    np.testing.assert_array_equal(elig,[False,False,True])
    assert s[0]==0.


def test_invalid_shape_and_nonfinite_predictions_fail_before_full_run():
    a=make_arr()
    a.pred[0,0,0]=np.inf
    with pytest.raises(RuntimeError,match='nonfinite'):
        audit_forecast_arrays(a,split='TRAIN')
    a=make_arr()
    a.observed=a.observed[:,:,:-1]
    with pytest.raises(RuntimeError,match='observed shape'):
        audit_forecast_arrays(a,split='VAL')


def test_horizon_cannot_use_unobserved_future_source():
    a=make_arr()
    with pytest.raises(ValueError,match='Invalid horizon'):
        edge_horizon_features(a,0,1,1,2)
