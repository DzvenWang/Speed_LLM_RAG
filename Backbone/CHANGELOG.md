# Changelog

## v0.5.0 — Shared Experiment Contract rerun release

- Added ExperimentCommon v1.0.0 as the single owner of materialized data, temporal features, cache identity, metrics, and prediction contract.
- Kept all five numerical model architectures and pinned official source commits unchanged.
- Unified Backbone aggregate metrics with Baseline using point-level micro MAE/RMSE; per-query metrics remain diagnostic only.
- Made PatchSTG the explicit configured downstream reference model; formal training no longer depends on a Top-2 shortlist artifact.
- Formal PatchSTG stability remains seeds 10/20/30, with one configured reference seed for all downstream consumers.
- Backbone remains the sole owner of `FROZEN_BACKBONE.json`.
- Added canonical prediction manifests and stronger current-code-version checks in screening/formal summaries.
- Removed historical migration/archive workflow from the main execution path.
