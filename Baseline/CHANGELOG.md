# Changelog

## v0.6.0 — Shared Experiment Contract rerun release

- Added ExperimentCommon v1.0.0 as the single data/time/metrics/prediction contract used by Backbone and Baseline.
- Removed duplicate Baseline materialization of the TraffiDent benchmark; project cache paths now attach to the shared ExperimentCache.
- Unified reported aggregate MAE/RMSE with Backbone using point-level micro aggregation.
- Made Baseline strictly read-only with respect to Backbone freeze artifacts.
- All PatchSTG-dependent Baseline seeds consume one reference checkpoint rather than seed-matched Backbone checkpoints.
- Added canonical prediction manifests and current-code-version guards to formal selection.
- Removed old cache migration/archive compatibility from the primary workflow.

## 0.6.0 implementation revision 20260913_rast_shim_leaf_tfp_r3
- Added RAST optional STID/AGCRN import compatibility shim outside frozen vendor source.
- Added RAST FAISS runtime/setup checks and dedicated development resubmission.
- Reworked LEAF to use the shared TFP Python runtime; removed isolated-env requirement.
- Added external TraffiDent GraphBranch/HypergraphBranch runner and official-style post-hoc selector JSON contract.
- Added LEAF development/formal/selector PBS scripts and runtime checks.
