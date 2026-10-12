# FARE v2.0.4 — Pilot Validation + Test and Retrieval Attribution

Based on v2.0.3. **No forecasting model architecture, training loss, checkpoint-selection or memory/retrieval mechanism has been modified.** This release changes *post-training evaluation* and its reports only.

## What changed

1. `FARE/pbs/train_pilot.pbs` now defaults to `EVAL_SPLITS=val:test` (previously `val`). `SEVERITIES=20:40:60:90` remains the default. The run trains once, selects `candidate_best.pt` using **Validation only**, then evaluates that same frozen checkpoint on Validation and all four Test severities in the same PBS GPU allocation.
2. `FARE/scripts/train.py` defaults to Val+Test for direct Pilot calls too. It runs five fixed-checkpoint Retrieval interventions per split: `full`, `similarity_memory`, `utility_off`, `random_memory`, `no_memory`. Test interventions are computed individually at s20/s40/s60/s90, not pooled into the Validation s40 data.
3. New `--ablation-splits val test` CLI or optional PBS variable `FARE_ABLATION_SPLITS=val:test`. Default is whichever splits are evaluated. `FARE_AUTO_ABLATIONS=0` disables the (potentially expensive) extra interventions but does not disable Val/Test forecast metrics.
4. Outputs `pilot_val_test_metrics.csv/.json/.md` plus `retrieval_attribution/val` and `retrieval_attribution/test` reports, including `retrieval_gain_val_test.md` with per-severity values and a clearly labelled unweighted Test severity mean.
5. `scripts/submit_gadi.py train --config FARE/configs/pilot.yaml` now also defaults to Val+Test when creating PBS jobs. Explicit `--eval-splits` still overrides.
6. Regression tests cover successful Val+Test, four Test severities, failure propagation, default Gadi submissions, attribution and reporting, with no actual Gadi or real traffic data required.

## Important evaluation protocol

- The *Validation* evaluator uses the frozen training-protocol Validation split (typically s40), not four separate Validation severities. The *Test* evaluator uses the four SupportBench evaluation caches (`s20/s40/s60/s90`).
- Pilot Test results are **exploratory only**. Do not use these results to select hyperparameters, change the model version, select the best stage, or choose checkpoints; otherwise the final Test becomes contaminated by repeated model selection. Preserve a separate untouched final evaluation or use an explicit development-Test protocol if repeated Test inspection is part of the research process.
- Fixed-checkpoint ablations are not equivalent to matched, separately trained models.
- Auto-ablations add 5 Validation and 5 × 4 Test inference evaluations at default severity settings and can materially increase Pilot walltime. Disable with `FARE_AUTO_ABLATIONS=0` when only forecast scores are needed, or restrict with `FARE_ABLATION_SPLITS=val`.

## Gadi submissions

```bash
qsub -N FARE_P204_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
qsub -N FARE_P204_flow  -v TARGET=flow,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
```

Explicit override, for pilot Test metrics but Validation-only attribution:

```bash
qsub -N FARE_P204_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32,EVAL_SPLITS=val:test,FARE_ABLATION_SPLITS=val FARE/pbs/train_pilot.pbs
```

## Expected output structure

```text
FARE_exps/<target>/<dataset>/pilot/<run>/
  candidate_best.pt
  posttrain_evaluation.json
  pilot_val_test_metrics.csv
  pilot_val_test_metrics.json
  pilot_val_test_metrics.md
  evaluation_candidate/
    val_full_summary.json
    s20/s20_full_summary.json
    s40/s40_full_summary.json
    s60/s60_full_summary.json
    s90/s90_full_summary.json
  retrieval_attribution/
    combined_ablation_summary.json
    retrieval_gain_val_test.md
    val/ablation_summary.csv
    val/retrieval_gain.md
    test/ablation_summary.csv
    test/retrieval_gain.md
```

No live Gadi speed, flow, or accuracy results are claimed by this code update.
