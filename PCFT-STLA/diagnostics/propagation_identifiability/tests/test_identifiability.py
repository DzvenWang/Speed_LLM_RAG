from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from diagnostic.stability import lag_pair_corr,compute_stability,change_and_valid
from diagnostic.gain import ForecastArrays,edge_horizon_features,gain_for_candidate,choose_unique_windows


def test_detect_known_leading_lag():
    rng=np.random.default_rng(4)
    T=1800
    src_delta=rng.normal(size=T).astype('f4')
    target_delta=np.roll(src_delta,3).copy()
    x=np.stack([np.cumsum(src_delta),np.cumsum(target_delta)],axis=1).astype('f4')
    valid=np.ones_like(x,bool)
    ei=np.array([[0,1],[1,0]],dtype=np.int64)
    dd,mm=change_and_valid(x,valid)
    r,n=lag_pair_corr(dd,mm,ei,3,min_pairs=10)
    assert r[0]>.99 and abs(r[1])<.15
    info=compute_stability(x,valid,ei,blocks=6,max_lag=6,min_pairs=10,
                           abs_corr=.15,lag_fraction=.66)
    assert info['best_lag'][0]==3
    assert info['stable_direction_lag'][0]
    assert not info['stable_direction_lag'][1]


def make_fake(R=600,known_lag=3,seed=10):
    rng=np.random.default_rng(seed)
    H,P,N=12,12,2
    x=rng.normal(size=(R,H,N)).astype('f4')
    obs=np.ones((R,H,N),dtype=bool)
    future_valid=np.ones((R,P,N),dtype=bool)
    pred=np.zeros((R,P,N),dtype='f4')
    y=np.zeros((R,P,N),dtype='f4')
    for h in range(1,known_lag+1):
        k=known_lag-h+1
        y[:,h-1,1]=2.0*(x[:,-k,0]-x[:,-k-1,0])+0.03*rng.normal(size=R)
    return ForecastArrays(np.arange(R,dtype=np.int64)*12,pred,y,future_valid,x,obs)


def test_gain_and_strict_timing():
    a=make_fake(600,3,1);b=make_fake(320,3,2)
    for h in (1,2,3):
        _,source,v=edge_horizon_features(a,0,1,3,h)
        assert v.all()
        k=3-h+1
        np.testing.assert_allclose(source,a.x[:,-k,0]-a.x[:,-k-1,0])
    gain=gain_for_candidate(a,b,0,1,3,min_fit=50,min_eval=20)
    assert gain['eligible'] and gain['gain_vs_local_pct']>80
    assert gain['gain_vs_stla_pct']>80
    assert gain['positive_groups']==4


def test_missing_source_no_information():
    a=make_fake(600,3,1);a.observed[:,:,-2:]=False  # remove target, not source
    a.observed[:,:,0]=False
    b=make_fake(320,3,2);b.observed[:,:,0]=False
    g=gain_for_candidate(a,b,0,1,3,min_fit=20,min_eval=10)
    assert not g['eligible']
