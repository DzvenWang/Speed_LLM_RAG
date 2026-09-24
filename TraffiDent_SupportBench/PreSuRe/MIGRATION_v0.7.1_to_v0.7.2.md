# PreSuRe v0.7.1 -> v0.7.2

## Why v0.7.2 exists

The v0.7.1 Development probe showed that exact future PatchSTG residual is not identifiable from the current deployment-safe state:

- exact-Need prediction was no better than a zero residual predictor;
- sign accuracy was approximately random;
- predicted Need did not improve support-utility ranking;
- true privileged Need made support utility almost deterministic.

Therefore v0.7.2 does **not** implement the originally proposed exact-residual v0.8. It first tests a weaker and more defensible hypothesis: whether a coarse **Predictive Need State** can be identified.

## New diagnostic tasks

All thresholds are derived from the Train `state_train` subset only, independently for every forecast horizon.

- `need_q75`: detect the top-25% absolute PatchSTG-error cells;
- `need_q90`: detect the top-10% absolute PatchSTG-error cells;
- `direction_q75`: positive/negative future base-error direction, evaluated only on q75 hard-error cells;
- `direction_q90`: direction on q90 hard-error cells;
- `magnitude_3`: small / medium / large error bands using Train q50/q80 thresholds;
- `signed_state_q75`: no-hard-need / positive-hard-need / negative-hard-need.

Each task has its own model instance. This is deliberate: the probe should measure identifiability rather than confound the result with multi-task gradient conflict.

## State-to-support bridge

The probe additionally evaluates three Top-4 support scores:

- `M3_CONF`: existing M3 candidate confidence, broadcast across horizons;
- `PREDICTED_STATE_ANALYTIC`: predicted need probability + direction + magnitude band are converted into a coarse residual-need estimate and scored against historical residuals using the same alpha-grid utility definition;
- `TRUE_STATE_ANALYTIC`: privileged coarse-state ceiling using true q75 need/direction and Train-derived magnitude-band centers. It still does not use the exact future residual magnitude.

This determines whether coarse need states contain enough information for support selection.

## Decision states

The new development gate is one of:

- `GO_V08_STATE`
- `CONDITIONAL_V08_STATE`
- `NO_GO_V08_STATE`

The exact-residual route is retained in the report as `NO_GO_FROM_V071`; v0.7.2 only decides whether a state-based v0.8 is justified.

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_static.py
python tests/smoke_predictive_need_state_probe.py
python scripts/state_probe_preflight.py --stage development
bash scripts/submit_predictive_need_state_probe.sh
```

The job reuses the completed v0.7 M1/M3 artifacts, PatchSTG predictions/residual memory and retrieval candidates. It does not rerun Oracle A-F or M0-M5.

## Outputs

Results are written under:

`PreSuRe_exps/PredictiveNeedStateProbe/TraffiDent_Net_06/development/<run>/`

Important files:

- `train_only_state_thresholds.json`
- `need_q75_summary.json`
- `need_q90_summary.json`
- `direction_q75_summary.json`
- `direction_q90_summary.json`
- `magnitude_3_summary.json`
- `signed_state_q75_summary.json`
- `hard_error_group_summary.json`
- `state_analytic_utility_summary.json`
- `state_probe_summary.csv`
- `summary.json`
- `DECISION.txt`

## PBS hotfix included

v0.7.2 also adds the missing `LOGROOT` definition to `pbs/predictive_need_probe.pbs`, fixing the v0.7.1 launch-time `LOGROOT: unbound variable` failure.
