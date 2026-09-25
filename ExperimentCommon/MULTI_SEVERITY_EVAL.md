# Multi-Severity Evaluation Cache Policy — v1.1.1

Four frozen paired Protocol evaluation manifests are materialized immediately:

- `protocol_core5_s20/evaluation`
- `protocol_core5_s40/evaluation`
- `protocol_core5_s60/evaluation`
- `protocol_core5_s90/evaluation`

Each contains 5,000 paired Test cases: 1,000 per main missing family.

`build_caches.py` deliberately uses a preparation-only bypass of Protocol's consumer gate. This does **not** grant model access. Cache manifests record `evaluation_access_at_materialization` and `evaluation_cache_materialized_while_locked`.

Because tensors exist on disk, researchers must treat the evaluation lock as procedural rather than physical. Standard model runners should enforce `require_evaluation_cache_access()` before opening Test arrays.
