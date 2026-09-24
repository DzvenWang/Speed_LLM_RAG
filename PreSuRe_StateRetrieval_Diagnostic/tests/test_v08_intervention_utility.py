import numpy as np
import pandas as pd

from srd.intervention_utility import (
    add_v08_features, add_utility_targets, feature_columns_v08,
    make_v08_model, fit_v08_model, predict_v08_scores,
)


def _frame(n=160):
    rng=np.random.default_rng(7)
    q=np.repeat(np.arange(max(1,n//8)),8)[:n]
    df=pd.DataFrame({
        'query_id':q,'query_pos':q,'patch_id':np.arange(n)%4,'recent_steps':np.where(np.arange(n)%2,12,6),
        'rank':np.arange(n)%20+1,'score':rng.random(n),'level_mae':rng.random(n),'delta_mae':rng.random(n),
        'overlap_fraction':rng.random(n),'tod_distance':rng.random(n),'dow_match':rng.integers(0,2,n),
        'patch_restorable_count':rng.integers(1,12,n),'patch_observed_count':rng.integers(10,40,n),
        'query_patch_observed_fraction':rng.random(n),'query_patch_missing_fraction_scope':rng.random(n),
        'query_patch_obs_mean':rng.normal(size=n),'query_patch_obs_std':np.abs(rng.normal(size=n))+0.1,
        'query_patch_recent_mean':rng.normal(size=n),'query_patch_recent_std':np.abs(rng.normal(size=n))+0.1,
        'query_patch_delta_abs_mean':np.abs(rng.normal(size=n)),
        'candidate_patch_mean':rng.normal(size=n),'candidate_patch_std':np.abs(rng.normal(size=n)),
        'candidate_patch_recent_mean':rng.normal(size=n),'candidate_patch_recent_std':np.abs(rng.normal(size=n)),
        'candidate_patch_delta_abs_mean':np.abs(rng.normal(size=n)),'candidate_query_mean_gap':rng.random(n),
        'candidate_query_std_gap':rng.random(n),'candidate_query_delta_gap':rng.random(n),
        'candidate_neighbor_observed_mae':rng.random(n),'affine_scale':rng.uniform(.5,2,n),'affine_shift':rng.normal(size=n),
        'affine_scale_deviation':rng.random(n),'affine_overlap_points':rng.integers(4,80,n),'affine_at_clip':rng.integers(0,2,n),
        'proposed_fill_count':rng.integers(1,40,n),'proposal_fill_mean':rng.normal(size=n),'proposal_fill_std':np.abs(rng.normal(size=n)),
        'proposal_abs_change_mean':rng.random(n),'proposal_abs_change_max':rng.random(n)*2,'proposal_change_rms':rng.random(n),
        'proposal_change_l2':rng.random(n)*4,'proposal_anchor_jump_mean':rng.random(n),'proposal_anchor_jump_max':rng.random(n)*2,
        'proposal_patch_after_mean':rng.normal(size=n),'proposal_patch_after_std':np.abs(rng.normal(size=n)),
        'proposal_obs_mean_gap':rng.random(n),'proposal_neighbor_gap_mean':rng.random(n),'proposal_neighbor_gap_max':rng.random(n)*2,
        'proposal_neighbor_support_fraction':rng.random(n),'prediction_shift_mae':rng.random(n),'prediction_shift_rmse':rng.random(n),
        'prediction_shift_max':rng.random(n)*3,'prediction_shift_std':rng.random(n),'prediction_shift_signed_mean':rng.normal(scale=.1,size=n),
        'prediction_patch_shift_mae':rng.random(n),'prediction_outside_patch_shift_mae':rng.random(n),'prediction_spill_ratio':rng.random(n)*2,
        'latent_global_mae':rng.random(n),'latent_global_rmse':rng.random(n),'latent_global_cosine':rng.uniform(.7,1,n),
        'latent_patch_mae':rng.random(n),'latent_patch_rmse':rng.random(n),'latent_patch_cosine':rng.uniform(.7,1,n),
        'latent_outside_patch_mae':rng.random(n),'latent_spill_ratio':rng.random(n)*2,
        'baseline_forecast_mae':rng.uniform(1.4,2.0,n),
    })
    # utility has a learnable intervention-aware signal with asymmetric harmful tails
    u=.025 - .05*df['proposal_abs_change_mean'].to_numpy() - .03*df['prediction_spill_ratio'].to_numpy() + rng.normal(0,.01,n)
    u[np.arange(n)%11==0]-=.08
    df['forecast_utility']=u
    df['forecast_mae']=df['baseline_forecast_mae']-df['forecast_utility']
    df['forecast_gain_pct']=100*df['forecast_utility']/df['baseline_forecast_mae']
    df['patch_completion_mae']=rng.random(n)
    return add_utility_targets(add_v08_features(df))


def test_v08_features_include_intervention_and_exclude_labels():
    df=_frame(80)
    cols=feature_columns_v08(df,'full')
    assert 'proposal_abs_change_mean' in cols
    assert 'latent_patch_mae' in cols
    assert 'prediction_shift_mae' in cols
    for bad in ['forecast_utility','forecast_mae','baseline_forecast_mae','patch_completion_mae','beneficial','harmful','utility_target']:
        assert bad not in cols


def test_v08_risk_models_fit_and_score():
    df=_frame(160);features=feature_columns_v08(df,'full')
    for name in ['logistic_intervention','cost_sensitive_gate','histgb_mean_intervention','histgb_q20_intervention','risk_expected_intervention']:
        spec=fit_v08_model(make_v08_model(name,10),df,features)
        score=predict_v08_scores(spec,df.iloc[:17],features)
        assert score.shape==(17,)
        assert np.isfinite(score).all()


def test_proposal_feature_set_omits_deep_shift_features():
    df=_frame(80)
    cols=feature_columns_v08(df,'proposal')
    assert 'proposal_abs_change_mean' in cols
    assert 'prediction_shift_mae' not in cols
    assert 'latent_patch_mae' not in cols
