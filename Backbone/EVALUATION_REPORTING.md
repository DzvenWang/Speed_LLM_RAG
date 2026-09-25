# Evaluation reporting v1.0.8

Evaluation is reported at two levels.

1. **Severity level**: s20, s40, s60, s90.
2. **Scenario level** within every severity, using the Protocol/SupportBench `family` field:
   - entry_random
   - sensor_outage
   - temporal_outage
   - regional_outage
   - directional_outage

The final tables report MAE, RMSE, directly-masked MAE (`Masked_MAE`), support-dependent MAE (`Support_MAE`), H12 MAE, and H12 support-dependent MAE.

Per checkpoint/severity output now includes:

- `test_metrics.csv` — per-query diagnostic metrics.
- `test_scenario_metrics.csv` — micro-aggregated metrics by scenario.
- `test_summary.json` — overall metrics plus `severity_summary` and `scenario_summary`.

The multi-severity checkpoint runner also writes:

- `Backbone_logs/PatchSTG.checkpoint_eval.all.seed<seed>.<stamp>.severity_summary.csv`
- `Backbone_logs/PatchSTG.checkpoint_eval.all.seed<seed>.<stamp>.scenario_summary.csv`

## Local explicit-checkpoint evaluation

```powershell
python Backbone\scripts\run_checkpoint_evaluations.py --checkpoint Backbone_exps\PatchSTG\TraffiDent_Net_06\formal\20260913_150736_seed10\best.pt --expected-seed 10
```

## GADI explicit-checkpoint evaluation

Default: one combined job, one live log, one final set of tables.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
bash submit_checkpoint_evaluations.sh \
  Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt 10
```

For four independent jobs instead:

```bash
PARALLEL=1 bash submit_checkpoint_evaluations.sh \
  Backbone_exps/PatchSTG/TraffiDent_Net_06/formal/20260913_150736_seed10/best.pt 10
```

## Summarize an existing checkpoint evaluation without rerunning inference

For the already-completed 2026-09-25 17:39 run:

```powershell
python Backbone\scripts\summarize_checkpoint_evaluation.py `
  --checkpoint-run-tag 20260913_150736_seed10 `
  --seed 10 `
  --run-stamp 20260925_173920
```

This reads the existing `test_summary.json`/`test_metrics.csv` files and creates severity/scenario summary CSVs without touching the model.
