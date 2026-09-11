# STDN_Special — attributable STDN diagnostic experiments

This directory is intentionally isolated from the main `Backbone/` code. It reuses the frozen official STDN source, the existing TraffiDent cache, `backbone.runner`, adapters and evaluation pipeline. **No file in `vendor/STDN` is edited**, so `SOURCE_LOCK` remains valid.

## Why this exists

The current STDN screening run reaches its best validation MAE very early (`epoch 3`, `val_MAE=2.231304`) and then overfits while train MAE continues to improve. The goal here is not a large hyperparameter search. The goal is to identify whether the large performance gap comes from optimization budget / regularization or from STDN's model-data fit.

## Experiment matrix

| ID | Batch | LR | Weight decay | Patience | What changes |
|---|---:|---:|---:|---:|---|
| official_control | 64 | 1e-3 | 0 | 10 | Existing result; do not rerun by default |
| d1_bs16 | 16 | 1e-3 | 0 | 10 | **Batch only** |
| d2_bs16_pat30 | 16 | 1e-3 | 0 | 30 | D1 + **patience only** |
| d3_regularized | 16 | 5e-4 | 1e-4 | 30 | D2 + **LR/regularization** |
| d4_low_lr | 16 | 2e-4 | 1e-4 | 30 | LR sensitivity check |

All diagnostics should remain on `screening`, seed 10, using the same frozen protocol/cache. Do not move to formal until the cause is understood.

## 0. Static checks

From `Backbone/`:

```bash
conda activate TFP
python STDN_Special/inspect_cache.py --stage screening
python -m py_compile STDN_Special/*.py
bash -n STDN_Special/submit_diagnostics.sh STDN_Special/pbs/*.pbs
```

## 1. Probe the real STDN batch first

```bash
qsub STDN_Special/pbs/probe_batch.pbs
```

The probe intentionally stops before the first optimizer update and writes:

```text
STDN_Special/artifacts/batch_probe_screening_seed10.json
```

Check the report for:

- integer temporal encoding candidates: DOW should be `0..6`, TOD should be `0..287` for 5-minute data;
- input tensor shape should correspond to H=12 and N=388;
- mask candidates and zero fractions;
- whether the batch contains both incomplete and full/clean history tensors.

**Do not submit tuning runs if the probe shows obviously wrong TE or missing-value encoding.** Fix the adapter first.

## 2. Run attributable diagnostics

Submit all four:

```bash
bash STDN_Special/submit_diagnostics.sh
```

Or one at a time:

```bash
qsub -v EXP=d1_bs16,STAGE=screening,SEED=10 STDN_Special/pbs/run_single.pbs
qsub -v EXP=d2_bs16_pat30,STAGE=screening,SEED=10 STDN_Special/pbs/run_single.pbs
qsub -v EXP=d3_regularized,STAGE=screening,SEED=10 STDN_Special/pbs/run_single.pbs
qsub -v EXP=d4_low_lr,STAGE=screening,SEED=10 STDN_Special/pbs/run_single.pbs
```

Live logs are written under `STDN_Special/logs/`.

## 3. Summarize results

```bash
python STDN_Special/analyze_logs.py STDN_Special/logs
```

Outputs:

```text
STDN_Special/artifacts/diagnostic_summary.csv
STDN_Special/artifacts/diagnostic_summary.md
```

Interpretation:

- **D1 strongly improves**: the original batch/update budget was a major factor.
- **D2 improves over D1**: early stopping/patience was truncating learning.
- **D3 improves materially**: learning rate / regularization mismatch was important.
- **D4 improves over D3**: STDN is still LR-sensitive on TraffiDent.
- **All remain around 2.1–2.2 after the adapter probe passes**: the remaining gap is much more plausibly model-setting mismatch rather than a trivial training configuration error.

## Complete-sensing control

A complete-sensing training control is intentionally **not fabricated** here. Whether it can be performed safely depends on what the real Backbone batch/cache contains. Run `probe_batch.py` first. If the probe confirms an explicit full/clean history tensor in the same batch, add a complete-sensing wrapper by replacing only the incomplete-history field at runtime. If it does not, build a separately frozen complete-sensing cache through the existing data-preparation pipeline rather than reconstructing inputs heuristically.

This restriction prevents an apparently useful control from accidentally changing sample selection, normalization, timestamps, or targets.

### Optional: complete-sensing STDN control after the probe

If and only if `batch_probe_screening_seed10.json` shows a matched full/clean history tensor with exactly the same shape/dtype as the incomplete input, submit:

```bash
qsub -v INPUT_KEY='<incomplete_history_key>',FULL_KEY='<full_history_key>' \
  STDN_Special/pbs/run_complete_sensing.pbs
```

The wrapper performs an exact shape/dtype check, changes only that one batch field at the STDN adapter boundary, and writes an audit JSON under `STDN_Special/artifacts/`. It refuses to run when those conditions are not satisfied.
