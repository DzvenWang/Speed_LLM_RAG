# Changelog

## 1.2.3
- Corrected Chapter-3 predictive-context metric semantics: formal `C_q^loss` is now `network_mean_context_loss = mean(per_target_context_loss)`.
- Retained v1.2.2 `aggregate_context_loss` as a backward-compatible q-quantile diagnostic only.
- Updated SO-v2 Stage A, support utility, forecast evaluation, legacy SO diagnostic and explanation evidence to use the formal network-mean metric.
- Corrected predictive-layout source contribution normalisation to average over all N targets; this is a uniform scale correction and does not change existing layout identities.
- Reworked predictive structured-mask audit to use the canonical API metric directly and to support OOD auditing without requiring an OOD SO-feature CSV.
- Added metric-semantics smoke tests and migration documentation.

## 1.2.2
- Replaced centrality-centre+BFS predictive layouts with direct Train-only dependency-context-loss optimisation. Regional layouts use connected greedy expansion; directional layouts greedily compose high-utility direction groups/nodes.
- Increased predictive-context audit sample budget from 64 to 512 matched Validation views (or all available views when fewer).
- Added multi-granularity backbone-conditioned support utility: all-node, directly-masked, support-dependent, H3/H6/H12, and globally safe support labels.
- Kept `forecast_gain` as a backward-compatible alias of global gain while making `gain_support_dependent` the default reasoner supervision.
- Added a separate large I1 training mining pool (default K=256) with multi-view hard candidates plus temporally diverse legal Train-memory negatives.
- Added `analyze_rag_headroom.py` for Beneficial Recall@K, safe-support recall, harmful Top-1 rate and oracle selection headroom.
- Physically separated explanation generator inputs from evaluator-only outcome labels; contrastive explanation inputs/labels are separated too.
- Extended reasoning pairs/records with support-dependent utility, global-safety signals and explicit hard-negative types.

## 1.2.1
- Fixed predictive-dependency lag semantics: `sample_stride` now thins correlation anchor times only; lag remains on raw traffic steps.
- With 5-min data and `max_lag_steps=3`, tested source-leading-target lags are 0/5/10/15 minutes, not 0/60/120/180 minutes.
- Added `lag_minutes`, `lag_step_minutes`, and sampling-cadence metadata to `predictive_dependency.npz/json`.
- Added build-audit checks that fail stale/incorrect dependency lag semantics.
- Added matched-Validation-window predictive-context diagnostics for general vs predictive regional/directional masks; low gain warns by default and can be made strict in config.
- Added explicit lag-minutes fields to LLM reasoning/explanation evidence.
- Added a dedicated dependency lag smoke test.

## 1.2.0
- Added General / Main Support / Support Challenge benchmark tracks.
- Added Train-only predictive-dependency graph and support-dependent target definition.
- Added predictive regional/directional sensing layouts derived only from Train dependency centrality.
- Main memory stride changed from 12 to 6; full and development memory budgets are explicit.
- Added temporal/spatial/propagation multi-view control retrieval with feature caching.
- Added Validation-calibrated future-free Support Opportunity score and exact High-Support-Opportunity query lists.
- Added all-node, directly-masked, support-dependent and H3/H6/H12 evaluation helper.
- Added stronger pairwise hard negatives and contrastive explanation evidence.
- Preserved v1.1.1 matched-severity OOD/adaptive layout catalogue safeguards.

## 1.1.1
- Adaptive feasible layout catalogue counts and OOD generation only at configured OOD severities.

## 1.1.0
- Dense canonical windows, matched-severity OOD layouts, stratified audit, resolved protocols and structured reasoning evidence.
