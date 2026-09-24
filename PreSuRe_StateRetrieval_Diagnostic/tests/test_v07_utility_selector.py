import numpy as np
import pandas as pd

from srd.utility_selector import (
    add_inference_features, feature_columns, expanding_time_folds,
    make_model, fit_model, predict_scores, tune_gate_and_cap, apply_decision_rule,
)


def synthetic():
    rng=np.random.default_rng(10); rows=[]
    for qpos,q in enumerate(range(100,160)):
        for L in [3,12]:
            for p in range(3):
                for r in range(1,6):
                    level=0.05+0.03*r+rng.normal(0,.005); delta=0.08+0.02*r+rng.normal(0,.005)
                    # Benefit is predictable but not identical to similarity rank.
                    u=0.06*(0.15-level)+0.025*(p==1)+0.02*(L==12)-0.01*(r==1 and p==2)+rng.normal(0,.004)
                    rows.append({
                        'query_id':q,'query_pos':qpos,'recent_steps':L,'patch_id':p,'rank':r,'candidate_id':1000+q*20+p*5+r,
                        'score':.75*level+.25*delta,'level_mae':level,'delta_mae':delta,'overlap_fraction':1.,'tod_distance':.1,'dow_match':1.,
                        'patch_restorable_count':20,'patch_observed_count':16,'query_patch_observed_fraction':.7,'query_patch_missing_fraction_scope':.3,
                        'query_patch_obs_mean':1.,'query_patch_obs_std':.2,'query_patch_recent_mean':1.05,'query_patch_recent_std':.2,'query_patch_delta_abs_mean':.08,
                        'baseline_forecast_mae':1.8,'forecast_mae':1.8-u,'forecast_utility':u,'forecast_gain_pct':u/1.8*100,'patch_completion_mae':.3,
                    })
    d=add_inference_features(pd.DataFrame(rows)); d['beneficial']=(d.forecast_utility>0).astype(int); d['harmful']=(d.forecast_utility<-.01).astype(int)
    return d


def test_v07_features_exclude_labels_and_hidden_completion():
    d=synthetic(); f=feature_columns(d)
    forbidden={'forecast_utility','forecast_mae','baseline_forecast_mae','patch_completion_mae','beneficial','harmful','candidate_id'}
    assert not forbidden.intersection(f)
    assert 'level_mae' in f and 'query_patch_observed_fraction' in f


def test_expanding_cv_is_strictly_causal():
    q=list(range(30)); folds=list(expanding_time_folds(q,6)); assert len(folds)>=3
    for _,fit,cal,test in folds:
        assert max(fit)<min(cal)<min(test)
        assert max(cal)<min(test)


def test_gate_reranker_runs_and_selects_distinct_patches():
    d=synthetic(); feats=feature_columns(d); qorder=d[['query_pos','query_id']].drop_duplicates().sort_values('query_pos').query_id.tolist()
    fold,fitq,calq,testq=next(iter(expanding_time_folds(qorder,6)))
    fit=d[d.query_id.isin(fitq)]; cal=d[d.query_id.isin(calq)]; test=d[d.query_id.isin(testq)]
    model=fit_model(make_model('histgb_utility',10),fit,feats)
    cs=predict_scores(model,cal,feats); ts=predict_scores(model,test,feats)
    gate=tune_gate_and_cap(cal,cs,5,[1,2,3])
    sel=apply_decision_rule(test,ts,5,gate['threshold'],gate['max_patches'])
    assert gate['max_patches'] in [1,2,3]
    if len(sel):
        assert not sel.duplicated(['query_id','recent_steps','patch_id']).any()
        assert sel.groupby(['query_id','recent_steps']).size().max()<=gate['max_patches']


def test_scope_missing_query_uses_fold_level_baseline_map():
    # q=101 has no recent1 action row, but it is still a fold test query and
    # must retain its baseline in an end-to-end recent1 evaluation.
    d=pd.DataFrame([
        {'query_id':100,'recent_steps':1,'baseline_forecast_mae':1.5},
        {'query_id':100,'recent_steps':12,'baseline_forecast_mae':1.5},
        {'query_id':101,'recent_steps':12,'baseline_forecast_mae':1.7},
    ])
    test_q=[100,101]
    testL=d[d.recent_steps==1]
    assert 101 not in testL.groupby('query_id').baseline_forecast_mae.first().index
    base_map=d.groupby('query_id',sort=False).baseline_forecast_mae.first()
    assert all(q in base_map.index for q in test_q)
    assert float(base_map.loc[101])==1.7


def test_empty_scope_action_tables_fall_back_to_abstention():
    d=synthetic().iloc[0:0].copy()
    scores=np.empty((0,),dtype=float)
    gate=tune_gate_and_cap(d,scores,50,[1,2,4])
    assert np.isinf(gate['threshold'])
    assert gate['max_patches']==0
    sel=apply_decision_rule(d,scores,50,gate['threshold'],gate['max_patches'])
    assert sel.empty
