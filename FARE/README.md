# FARE v2.1.1 — Forecast-Oriented Retrieval with Shared Benchmark Evaluation

This is a **clean architectural implementation**, not an upgrade/compatibility
patch for FARE v1/v2.0. Old checkpoints, optimizer states, and model configs
are rejected. The implementation is *research software*: CPU synthetic tests
pass, but this release has **not** yet been trained or evaluated on Gadi or
verified against the user's installed Base_SupportBench/Backbone data.

## Research goal

The target is **predictive benefit attributable to correct retrieval**, not
merely stronger traffic imputation. STLAformer remains the strong forecast
backbone. Retrieval must be tested against random, similarity-only, cross-query
shuffled, null and no-memory controls on Speed and Flow.

## Architecture

1. `ForecastQueryEncoder` in `fare/support_query.py`: mask-aware, anchor-wise
   temporal encoding with directed upstream/downstream indicators; node query
   retains STLA hidden/forecast context. No early anchor averaging.
2. `HierarchicalForecastRetriever` in `fare/retrieval.py`: region-level
   historical memory Top-64, node-specific low-dimensional Top-16 reranking,
   utility Top-8 plus a trainable null candidate. Historical outcomes must be
   known before the query issue time. Candidate IDs and validity are separated.
3. `TrainMemory` in `fare/memory.py`: train-only historical states, validity,
   actual historical future, and teacher latent. Sparse node/history/future/
   latent gather avoids materializing a full network for every candidate.
4. `MaskConditionedEvidenceAdapter` in `fare/evidence_adapter.py`: adapts the
   actual historical source/target episode under the query observation mask,
   aligns observed levels and provides future trajectory evidence, predictive
   token, state token, residual and reliability. Actual historical future is
   used only for temporally completed memory episodes.
5. `CandidateForecastProbe`: a candidate-specific, trained prediction probe.
   Node candidate improvement relative to the frozen STLA forecast supervises
   the utility ranker, pairwise loss, ranking regret and coarse contrastive
   retrieval scores. No current target is passed into model forward.
6. `FARE` in `fare/model.py`: externally retrieved tokens are injected through
   STLA attention blocks (default layer 4/5). Node-specific future evidence is
   transferred via a forecast-space correction and controlled by a reliability
   gate. **No external evidence means exactly zero retrieval correction.**

### Training stages

| Stage | Primary trainable components | Selection |
|---|---|---|
| `evidence` | Historical evidence adapter and candidate forecast probe | Validation probe objective |
| `retrieval` | Query/key, fast reranker, utility ranking, null weight | Validation ranking objective |
| `integration` | Retrieval, evidence adaptation, STLA retrieval injection, forecast control | Validation forecast MAE |
| `control` | Selective forecast gate | Validation forecast MAE |

**Purged teacher holdout:** by default the evidence/probe stage uses the
chronologically earlier ~75% of TRAIN and the retrieval utility ranking stage
uses the later ~25%, with a 24-step purge around the split. This prevents the
candidate probe being directly trained on its utility-supervision queries.
It is a **single time-held-out teacher**, **not full K-fold OOF**. Integration
then trains on all selected TRAIN windows. `FARE_UTILITY_TEACHER_PARTITION`
records the actual counts, split boundary and purge gap.

The implemented masked historical replay is **a learned, query-conditioned
adapter**, not a fully precomputed two-backbone complete/masked replay teacher
bank. These are distinct designs; do not claim the latter has been built.

## Evaluation: mandatory common Base_SupportBench protocol

**FARE does not define its own MAE/RMSE/MAPE.** Deploy the three-project
`Traffic_ThreeProjects_MAPE_20261012(1).zip` update into the **repository
root** (all three updated `Base_SupportBench/`, `Backbone/`, `Baseline/`
directories). `FARE/fare/evaluation.py` imports the following *live shared*
interfaces:

```python
from experiment_common.metrics import (
    metric_bundle, per_query_metrics, MAPE_PROTOCOL, MAPE_MIN_TARGET_ABS
)
```

- Final, Base, Candidate, Family, severity and horizon scores use
  `metric_bundle(pred_raw, y_raw, future_valid, direct_mask)`.
- Headline MAE/RMSE/MAPE are **cell-micro-aggregated**. The `direct_mask`
  selects exactly the same directly-masked nodes as Backbone and Baseline.
- `H3`, `H6`, `H12` are **individual forecast steps** 3, 6, 12, not prefix means.
- MAPE is in percent, uses **raw target units** and the public strict
  `abs(y_raw) > 1.0` rule for both Speed and Flow; MAE/RMSE still include zeros.
- `MAPE_count`, `MAPE_coverage`, `count` and the corresponding masked and
  horizon fields are preserved. If no cell qualifies, shared MAPE is NaN,
  not a fabricated 0%; if no valid MAE cells exist coverage is NaN.
- `metrics_revision=traffident_metrics_v2` remains unchanged, and
  `mape_protocol=raw_target_percent_exclude_abs_le_1_v1` is written to outputs.
- The threshold is **not independently configurable in FARE**. To maintain
  benchmark comparability, FARE rejects mismatched/older shared protocols.
- Per-query CSV diagnostics call the shared `per_query_metrics()` function,
  but headline MAPE is **never averaged across those rows**.
- The four Test severities are reported individually, alongside a separate
  **equal-severity macro average**. The macro average is not pooled MAPE.

Run immediately after deploying the updated common project, without any dataset
cache or model checkpoint:

```bash
python -u FARE/scripts/check_metric_protocol.py
```

See `METRICS_INTEGRATION.md` for field mapping, outputs and tests.

Validation selects the checkpoint using the existing forecast MAE objective;
Pilot Test s20/s40/s60/s90 is **exploratory** and never selects checkpoints.
Because Test was repeatedly viewed during development, final paper claims
need independently controlled testing.

### Fixed-checkpoint retrieval interventions

`full`, `similarity_memory`, `utility_off`, `random_memory`,
`cross_query_shuffle`, `weight_shuffle`, `null_evidence`, `retrieval_off`,
`no_memory`.

`retrieval_off` is the untouched frozen STLA result. `no_memory` on a full
checkpoint also has no learned no-memory adapter. For an independently
retrained no-memory comparator, use `--variant no_memory` in a separate run.
Fixed-checkpoint interventions alone do **not** establish causal gains from
retrieval vs extra training capacity; a separately trained matched comparator
and multiple seeds remain necessary.

## Dependencies

An existing `Speed_LLM_RAG/` root must contain:
`Base_SupportBench`, its built speed/flow caches and resolved configs,
`Backbone/vendor/STLAformer/...` and pretrained STLAformer checkpoints.
The benchmark and backbone are intentionally not duplicated here. Requires
Python 3.11+, PyTorch, NumPy, pandas, PyYAML and the TFP environment.

## Gadi pilot workflow

Unzip **FARE/** under the repository root. Back up/rename the old `FARE/`
first. **Do not overwrite a running old PBS job's source tree.**

```bash
conda activate TFP
python -u FARE/scripts/preflight.py --target speed --config FARE/configs/pilot.yaml
python -u FARE/scripts/preflight.py --target flow --config FARE/configs/pilot.yaml

qsub -N FARE21_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
qsub -N FARE21_flow  -v TARGET=flow,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
```

PBS default resources: **1 V100, 12 CPUs, 32 GB RAM, 24h pilot / 48h formal**;
`gpuvolta` queue, project `mh20`. The initial pilot has 8,000 time-spread
training windows and 16,000 train-only unique-memory windows. Full Validation
is used unless otherwise configured. The formal config has a larger memory and
uses initial BS16 for GPU-memory safety.

Pilot performs Validation **and Test** auto-evaluation and, by default, all
nine attribution interventions on both. For initial architecture testing to
avoid expensive ablations set `FARE_AUTO_ABLATIONS=0` in the `qsub -v` list.

To run a small local integration smoke test (requires actual benchmark caches):

```bash
python -u FARE/scripts/train.py --config FARE/configs/pilot.yaml --target speed --smoke
```

For independent no-memory training:

```bash
python -u FARE/scripts/train.py --config FARE/configs/pilot.yaml --target speed \
  --variant no_memory --phases integration
```

For a completed run, the usual outputs include:

```text
FARE_exps/<target>/<dataset>/pilot/<stamp>_seed10/
  candidate_best.pt
  memory.npz
  phase_summary.json
  diagnostics/epoch_diagnostics.csv
  pilot_val_test_metrics.csv
  pilot_val_test_metrics.md
  evaluation_candidate/val_full_summary.json
  evaluation_candidate/s20/s20_full_summary.json
  ... s40, s60, s90
  retrieval_attribution/val/ablation_summary.csv
  retrieval_attribution/test/ablation_summary.csv
  retrieval_attribution/retrieval_gain_val_test.md
```

## CPU tests and deployment limitations

```bash
cd FARE
python -m pytest -q tests/test_v21_core.py tests/test_shared_mape_protocol.py
python -m compileall -q fare scripts
bash -n pbs/train_pilot.pbs
```

CPU synthetic regressions use the real shared MAPE implementation and cover complete forward, null-evidence/base equality,
future-label isolation, causal empty-memory behavior, stage gradients,
training/checkpoint loop, temporally purged teacher partition, zero-denominator
MAPE handling and multi-format report writing. They do **not** establish actual
V100 throughput, memory peak, convergence or predictive benefit. On Gadi,
inspect `FARE_RESOURCE_EPOCH`, `FARE_DIAG`, `FARE_EPOCH`, `FARE_STAGE_DIAG`
and evaluate Real vs Random/Shuffled rather than assuming retrieval succeeds.
