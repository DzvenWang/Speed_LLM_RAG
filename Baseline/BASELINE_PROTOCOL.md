# Baseline Protocol v0.4.0

## Experimental question chain

The main table is not a flat collection of SOTA methods. Each row rejects a simpler explanation for gains under incomplete sensing:

`Backbone -> Reconstruction-first -> Missing-aware -> Generic Retrieval -> Traffic Retrieval -> Robust Retrieval -> Predictive-Support Reasoning`.

## Non-negotiable fairness rules

1. Consume frozen Shared Protocol view identities only; no local resampling.
2. Use Train/Val only before evaluation authorization.
3. Select checkpoints by Val only.
4. Retrieval memory is Train only.
5. Historical memory may use complete Train history; query history remains exactly the frozen incomplete observation.
6. RAFT/SARAF exclude any Train-memory window whose `[history + future]` time interval overlaps a Train or Val query interval.
7. ImputeFormer and matched RAST comparisons that rely on a forecasting backbone wait for `FROZEN_BACKBONE.json`.
8. SLPF is reported only under a matched fixed partial-sensing subtrack, not as a native dynamic-mask main-table method.
9. LEAF is isolated as an LLM-framework comparison; its legacy/vLLM environment must not mutate TFP.

## Metrics

All methods export raw-speed predictions with shape `[M, 12, N]`, aligned by `view_id`. The common evaluator reports Overall/Masked/Support MAE and RMSE plus H3/H6/H12 MAE and H3/H6/H12 Support MAE.
