# ExperimentCommon v1.1.1

- Default `build_caches.py` now materializes development, formal, and all s20/s40/s60/s90 evaluation caches.
- Added preparation-only locked-evaluation materialization/validation flags.
- Evaluation cache manifests explicitly record whether materialization happened while access was locked.
- Added `require_evaluation_cache_access()` for downstream model consumers.
- Added `--training-only` to preserve the previous training-cache-only workflow.
- Tensor, metric, prediction, and time-feature semantics are unchanged from v1.1.0.
