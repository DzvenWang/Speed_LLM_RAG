# Baseline Protocol v0.6.0

Baseline v0.6.0 consumes the frozen Shared Protocol exclusively and shares `ExperimentCommon v1.0.0` with Backbone v0.5.0.

Main guarantees:

1. No Baseline-local resampling of development/formal/evaluation views.
2. One common materialized cache per protocol stage.
3. One canonical TOD/DOW source and fingerprint.
4. One point-level micro MAE/RMSE definition for all cross-method tables.
5. Raw-speed prediction arrays use `[M, P, N]` and explicit `view_id` alignment.
6. Retrieval memory is train-only and the current query future is unavailable at inference.
7. PatchSTG-dependent methods use the single Backbone-owned reference checkpoint for every Baseline seed.
8. Test/evaluation data remain isolated until the formal selection/protocol freeze authorizes evaluation.
