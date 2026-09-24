# v0.7 implementation notes

## Research target

v0.6 showed that the same patch-local Top-50 pool has a large Forecast-Utility Oracle but poor similarity/completion-based realizability. v0.7 therefore keeps candidate generation fixed and learns the missing decision layer:

1. **Support gate** — should this damaged patch be modified at all?
2. **Candidate reranker** — which Top-50 historical support has the highest expected downstream utility?
3. **Patch-count control** — how many high-confidence patch interventions should be executed together?

## Labels

Each query-patch-candidate action uses the v0.6 diagnostic label

`forecast_utility = baseline PatchSTG MAE - single-patch-supported PatchSTG MAE`.

The label is allowed only for historical fit/calibration blocks and evaluation. It is not a test-time feature.

## Primary protocol

Expanding temporal CV is the default. Every fold obeys:

`fit time < calibration time < test time`.

The calibration block chooses the abstention threshold and patch-count cap. Test decisions are then frozen and evaluated by actually composing selected supports and re-running frozen PatchSTG.

## Model suite

- similarity_rule — retrieval-distance control baseline;
- logistic_gate — predicts probability of positive utility;
- ridge_utility — linear utility regression;
- histgb_utility — nonlinear utility regression;
- two_stage_histgb — benefit probability + positive-utility magnitude score.

## Main output

The primary number is `realized_gain_pct` in `predictive_utility_selector/model_comparison.csv`, especially `recent_steps=12` under `protocol=expanding`.
