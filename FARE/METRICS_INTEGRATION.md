# FARE v2.1.1 · Shared Evaluation Contract

Reference: `Traffic_ThreeProjects_MAPE_20261012(1).zip` and
`Base_SupportBench/MAPE_EVALUATION_PROTOCOL.md`.

## One public implementation, zero duplicated metric formulas

`fare/evaluation.py` imports from the deployed
`Base_SupportBench/experiment_common/metrics.py`:

- `metric_bundle(pred, true, valid, direct_mask)` — the ONLY evaluator used
  for Model Final/Base/Candidate and all family/severity scopes.
- `per_query_metrics(...)` — per-query CSV diagnostics, **not** for headline
  averaging.
- `MAPE_PROTOCOL` and `MAPE_MIN_TARGET_ABS` — mandatory protocol check.

FARE uses raw values `prediction * std + mean` and cached `y_raw`, with the
same `future_valid` and `direct_mask` as Backbone and Baseline. For all
methods, `metric_bundle()` returns canonical `overall_*`, `masked_*`,
`H3_*`, `H6_*`, `H12_*`, cell counts, MAPE-count/coverage and protocol metadata.
For consumers that expect simple names, FARE ADDITIVELY maps:

| FARE field | Shared bundle field |
|---|---|
| `MAE`, `RMSE`, `MAPE` | `overall_MAE`, `overall_RMSE`, `overall_MAPE` |
| `directly_masked_MAE/RMSE/MAPE` | `masked_MAE/RMSE/MAPE` |
| `MAPE_count`, `MAPE_coverage` | `overall_MAPE_count`, `overall_MAPE_coverage` |
| `directly_masked_MAPE_count/coverage` | `masked_MAPE_count/coverage` |
| `H3_MAPE`, `H6_MAPE`, `H12_MAPE` | `H3/H6/H12_overall_MAPE` |
| `H3/H6/H12_directly_masked_MAPE` | `H3/H6/H12_masked_MAPE` |

All canonical fields remain in the same result dictionary. FARE does not
replace, clip or recompute their values.

## Exact MAPE policy

- Native/raw target units; `abs(truth) > 1.0` (strict) and finite pred/truth.
- Same validity/scope masks as MAE/RMSE, but zero and low values are omitted
  **only** from MAPE.
- A score of `25.0` means 25 percent; not 0.25.
- Empty eligible cells: `MAPE=NaN`, `MAPE_count=0`. MAPE coverage is
  `MAPE_count / count` (NaN if count=0).
- H3/H6/H12 refer to forecast steps 3/6/12 individually.
- Within a severity, metric_bundle micro-aggregates cells; the separate
  four-severity `Avg(s20,s40,s60,s90)` is a macro average.

## Output files

The existing files now inherit ALL shared metric fields, including the new
horizon and directly-masked MAPE counts/coverage:

```
FARE_exps/<target>/<dataset>/pilot/<run>/
  stages/<phase>/val_summary.json
  stages/<phase>/val_per_query_metrics.csv       # when stage arrays enabled
  evaluation_candidate/val_full_summary.json
  evaluation_candidate/val_full_per_query_metrics.csv
  evaluation_candidate/s20/s20_full_summary.json
  evaluation_candidate/s20/s20_full_per_query_metrics.csv
  ... s40/s60/s90
  pilot_val_test_metrics.csv
  pilot_val_test_metrics.json
  pilot_val_test_metrics.md
  retrieval_attribution/val/ablation_summary.csv
  retrieval_attribution/test/ablation_summary.csv
  retrieval_attribution/retrieval_gain_val_test.md
```

Family reports (`*_family_metrics.csv`) and all retrieval modes use the
same public metric_bundle method. `pilot_val_test_metrics.csv` additionally
includes H3/H6/H12 and directly masked MAE/RMSE/MAPE for Base/Candidate/FARE.

## Deployment and sanity checks

1. Extract the THREE project MAPE update to repository root, then extract
   this FARE update to `FARE/`. No datasets/checkpoints need rebuilding just
   to add MAPE columns. Previously saved evaluation files need reevaluation.
2. In `conda activate TFP`, from repo root:

```bash
python -u FARE/scripts/check_metric_protocol.py
python -m pytest -q FARE/tests/test_v21_core.py FARE/tests/test_shared_mape_protocol.py
python -u FARE/scripts/preflight.py --target speed --config FARE/configs/pilot.yaml
python -u FARE/scripts/preflight.py --target flow --config FARE/configs/pilot.yaml
```

3. Use existing `FARE/scripts/evaluate.py` for selected checkpoints, e.g.:

```bash
python -u FARE/scripts/evaluate.py --checkpoint /path/to/candidate_best.pt \
  --split test --severities 20 40 60 90
```

No changes were made to STLAformer weights, the underlying Base_SupportBench
caches, training data, checkpoint selection method or retrieval architecture
by this evaluation-interface update.
