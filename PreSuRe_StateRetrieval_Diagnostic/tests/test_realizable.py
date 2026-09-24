import numpy as np
import pandas as pd

from srd.realizable import add_features, feature_columns, make_model, fit_model, predict_scores, tune_threshold, evaluate_decisions


def main():
    rng = np.random.default_rng(10)
    rows = []
    scopes = [1,2,3,4,6]
    for q in range(24):
        base = 2.0
        for r in range(1, 6):
            for L in scopes:
                recent3 = rng.uniform(0.1, 0.9)
                ctx = rng.uniform(0, 1)
                u = 0.08 * (0.55 - recent3) + 0.04 * (ctx - 0.5) - 0.01 * abs(L-4) + rng.normal(0, 0.01)
                rows.append({
                    'query_id': q, 'candidate_rank': r, 'candidate_id': q*10+r,
                    'scope_steps': L, 'retrieval_score': 1-recent3, 'latent_similarity': 0.9,
                    'observed_mae': recent3, 'observed_mae_recent3': recent3,
                    'observed_mae_recent6': recent3+0.02, 'observed_mae_all': recent3+0.04,
                    'context_score': ctx, 'overlap_fraction': .9, 'overlap_recent3': .9,
                    'overlap_recent6': .9, 'fill_count': 100+L,
                    'baseline_masked_mae': base, 'restored_masked_mae': base-u,
                    'utility': u, 'beneficial': int(u>0), 'harmful': int(u<-.01),
                })
    df = add_features(pd.DataFrame(rows))
    feats = feature_columns(df)
    forbidden = {'utility','beneficial','harmful','baseline_masked_mae','restored_masked_mae'}
    assert not forbidden.intersection(feats)
    fit = df[df.query_id < 14]
    calib = df[(df.query_id >= 14) & (df.query_id < 19)]
    test = df[df.query_id >= 19]
    spec = fit_model(make_model('ridge_utility', 10), fit, feats)
    cal_s = predict_scores(spec, calib, feats)
    tau, _ = tune_threshold(calib, cal_s)
    test_s = predict_scores(spec, test, feats)
    metrics, decisions = evaluate_decisions(test, test_s, tau, harmful_margin=.01)
    assert len(decisions) == 5
    assert np.isfinite(metrics['realized_gain_pct'])
    assert np.isfinite(metrics['oracle_gain_pct'])
    print('synthetic realizable benchmark tests PASS')


if __name__ == '__main__':
    main()
