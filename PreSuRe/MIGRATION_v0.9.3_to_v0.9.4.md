# PreSuRe v0.9.3 -> v0.9.4

v0.9A.3 showed that a Top-1 selective acceptance head is almost unable to determine whether the frozen v0.9A support has positive utility (Final masked AUROC approximately 0.545); the calibrated policy therefore degenerates to accepting nearly every Top-1 support. v0.9.4 does **not** add another acceptance head. It introduces a query-only diagnostic to test the missing information directly: can deployment-safe query context predict the sign of the future PatchSTG error?

## Query Future-Error Direction Probe

For every query/node/horizon,

```text
E* = Y_future - Y_base
```

`E* > 0` means PatchSTG underpredicts and a positive correction is required; `E* < 0` means PatchSTG overpredicts and a negative correction is required.

The model input contains only the existing PreSuRe query node features: incomplete normalized history, history observation mask, neighbour context, time covariates, explicit sensing-state descriptors, and the frozen PatchSTG forecast trajectory. It receives **no retrieval candidate, support residual, Oracle choice, Qwen output, direct/support evaluation mask, or current-query future target**.

Training uses the exact v0.9A `train_ids`; model selection uses its exact `dev_ids`; evaluation uses its exact `final_val_ids`. Horizon-specific q50/q75/q90 absolute-error thresholds are derived from the v0.9A Train split only. Direction CE is trained on errors at or above Train q50 to suppress near-zero sign noise, while a signed-error Huber head is trained on all valid cells. Final diagnostics report direction AUROC/balanced accuracy at q50/q75/q90, signed-error correlation, and masked horizon-by-horizon metrics.

Interpretation:

- `GO_RESIDUAL_ALIGNMENT`: the query contains sufficient correction-direction signal to justify a Query Correction Demand -> Support Residual Alignment module.
- `CONDITIONAL_ALIGNMENT`: weak but non-random direction signal exists; alignment should remain low-capacity and strongly gated.
- `NO_GO_QUERY_REPRESENTATION`: current query representation still lacks enough future correction information; revise the query encoder before adding more retrieval/acceptance logic.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_query_future_error_direction_probe.py
python scripts/query_future_error_direction_preflight.py --stage development
bash scripts/submit_query_future_error_direction_probe.sh
```

PBS job name: `QErrDir`.
