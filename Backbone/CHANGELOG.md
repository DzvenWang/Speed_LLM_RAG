# Changelog

## v0.6.0 — STLAformer TRC 2026 integration

- Added STLAformer as the sixth numerical backbone candidate using the official repository pinned at `f3eba25e2c63023a4834fd87d88da1c3347f16a9`.
- Added a thin STLAformer adapter that preserves the official `STLformer` architecture and maps TraffiDent canonical history to normalized traffic + fractional time-of-week. Paper-reported training settings are used for batch size (16), Adam LR (0.001), 200 epochs, 6 layers, and plateau LR decay (factor 0.1, patience 20).
- Added STLAformer to screening, PBS validation, source locking, preflight, static recent-model checks, and GPU adapter smoke support.
- Kept PatchSTG as the configured downstream `reference_model`; adding STLAformer does not silently change the frozen downstream reference.
- Added an explicit compatibility rule so current-contract v0.5.0 screening/formal evidence for the unchanged original models can be reused; STLAformer itself requires v0.6.0.
- No vendor model source is modified.

## v0.5.0 — Shared Experiment Contract rerun release

- Added ExperimentCommon v1.0.0 as the single owner of materialized data, temporal features, cache identity, metrics, and prediction contract.
- Kept all five numerical model architectures and pinned official source commits unchanged.
- Unified Backbone aggregate metrics with Baseline using point-level micro MAE/RMSE; per-query metrics remain diagnostic only.
- Made PatchSTG the explicit configured downstream reference model; formal training no longer depends on a Top-2 shortlist artifact.
- Formal PatchSTG stability remains seeds 10/20/30, with one configured reference seed for all downstream consumers.
- Backbone remains the sole owner of `FROZEN_BACKBONE.json`.
- Added canonical prediction manifests and stronger current-code-version checks in screening/formal summaries.
- Removed historical migration/archive workflow from the main execution path.
