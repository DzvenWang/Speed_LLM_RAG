# MAPE Evaluation Extension (Base_SupportBench + Backbone + Baseline)

## Definition and low/zero-valued targets

**MAPE (%) = 100 × mean(|prediction − truth| / |truth|)** over valid
forecast cells satisfying **|truth| > 1.0** in the **raw target unit** (not
standardized values). The same strict threshold applies to `speed` and `flow`
and to all severity levels, failure families, and prediction horizons.

* All other original validity conditions still apply: the same `future_valid`
  and scenario masks, and finite prediction/truth.
* **MAE and RMSE are unchanged**, and continue to include all original valid
  cells, including zeros.
* `masked_MAPE` uses the exact original directly-masked node selection.
* `H3`, `H6`, and `H12` use the *individual horizon-step* conventions of the
  existing MAE/RMSE protocol (not prefix averages).
* A scope with no eligible MAPE cells gets `NaN` and `MAPE_count=0`. Such
  cases **must not** be silently treated as zero percent.
* `MAPE_count` and `MAPE_coverage=MAPE_count/count` disclose how many valid
  targets were eligible for percentage error. An overall or family MAPE is
  micro-aggregated across eligible *cells*, not averaged across per-query MAPE.
* A value of `25.0` means **25%**, not `0.25`.

The threshold 1.0 is a deliberate reporting definition, not the result of an
experiment-tuned epsilon. It prevents division by zero and unstable inflation
from very small target values, especially for Flow. It also means MAPE covers
fewer cells than MAE/RMSE; compare MAPE only when exclusion thresholds and
coverage are stated. Other published papers may use incompatible MAPE masks.

## Identity/checkpoint compatibility

**`METRICS_REVISION=traffident_metrics_v2` is intentionally preserved.** It is
embedded into Base_SupportBench cache manifests and historical Backbone/
Baseline run identity fingerprints. This extension appends a new metric but
never changes cache schema, model training, best-checkpoint selection (MAE),
checkpoint format, normalization, train/validation/test splits or severity
sampling. To avoid invalidating frozen artifacts, the new evaluator separately
records `mape_protocol=raw_target_percent_exclude_abs_le_1_v1` and
`mape_min_target_abs=1.0` in metric bundles.

Previously saved evaluation CSV/JSON do **not** acquire MAPE automatically.
Run Evaluate using the new three-project code to compute new metric columns.
No training or cache rebuild is needed. Keep **all three versions in sync**;
only updating Backbone or Baseline without Base_SupportBench will fail due to
missing `MAPE` keys.

## Outputs

* Base_SupportBench `experiment_common.metrics.metric_bundle()` returns:
  `overall_MAPE`, `masked_MAPE`, `H3/H6/H12_{overall,masked}_MAPE`,
  `{scope}_MAPE_count`, `{scope}_MAPE_coverage`, while retaining MAE/RMSE.
* Backbone `test_metrics.csv`, `test_summary.json`, severity and scenario CSVs,
  live terminal tables and its per-query CSV get MAPE.
* Baseline `val_metrics.csv`, `test_metrics.csv`, summary JSON,
  `severity_summary.csv`, `scenario_summary.csv`, and live terminal tables
  get MAPE. Original MAE validation selection is unchanged.

### Backbone checkpoint Evaluate (example on Gadi)

```bash
python -u Backbone/scripts/post_training_evaluate.py \
  --model STLAformer --seed 10 \
  --checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal_complete/20261009_170304_seed10/best.pt
```

Set `TRAFFIDENT_TARGET=speed` before Speed Evaluation, or `flow` before Flow.
Original `formal` incomplete checkpoints work in the same command.

### Baseline re-evaluation

The normal Baseline evaluation scripts support this extension with no new CLI
flag. For example (within an environment with Gadi cache and model sources):

```bash
python -u Baseline/scripts/run_evaluations.py \
  --method RAFT-STLAformer --backbone STLAformer \
  --backbone-train-mode complete --seed 10
```

Baseline still trains on incomplete observations regardless of reference
Backbone training mode; the mode selects only which pretrained checkpoint is
loaded. The normal PBS wrappers continue to provide timestamped live logs.
