# Oracle_Diagnostic prediction export

This utility exports the frozen PatchSTG predictions required by `Oracle_Diagnostic`.
It is intentionally aligned with the current Core-5 evaluation path rather than the
legacy challenge-pool exporter.

Outputs are written by default to:

`Oracle_Diagnostic_exps/prediction_contracts/PatchSTG/<checkpoint-run>.<timestamp>/`

Files:

- `s20_evaluation_predictions.npz`
- `s40_evaluation_predictions.npz`
- `s60_evaluation_predictions.npz`
- `s90_evaluation_predictions.npz`
- `train_memory_predictions.npz`
- `export_summary.json`
- `oracle_prediction_env.sh`

The four query files contain `view_id` and `pred [5000,12,388]` in raw speed units.
The memory file contains `episode_id` and `pred [M,12,388]` in raw speed units.

The exporter always loads `Backbone_exps/FROZEN_BACKBONE.json`, verifies the reference
checkpoint SHA256 and formal identity, uses the **current SupportBench PatchSTG layout**
(the same policy as `scripts/evaluate_checkpoint.py`), and strictly loads the checkpoint
weights.  It does not retrain or modify any checkpoint/cache/protocol artifact.

## Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
qsub Backbone/pbs/export_oracle_predictions.pbs
```

Watch the live log:

```bash
tail -f Backbone_logs/OraclePredictionExport.*.live.log
```

At completion the log prints `OUTPUT_DIR` and `ORACLE_ENV_FILE`. Then:

```bash
source <OUTPUT_DIR>/oracle_prediction_env.sh
python Backbone/scripts/validate_oracle_prediction_export.py --output-dir <OUTPUT_DIR>
python Oracle_Diagnostic/scripts/preflight.py
```
