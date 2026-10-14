# FARE v3.0.2 — Traffic Propagation Support Restoration

**A new, standalone research implementation.** No FARE v1/v2 checkpoint or project dependency; only the durable `Backbone` official STLAformer and `Base_SupportBench` data/metric contracts are required.

FARE is designed for single-run full-model training plus **fixed-checkpoint internal attribution**. Neither external baseline comparisons nor retrained ablation models are part of the automated report.

## Implemented architecture

1. `TrafficGraph` + `PropagationSupportEncoder`: directional/undirected graph coverage, horizon-specific influence, exact contiguous missing components, observable boundary ratio and soft traffic-state features.
2. `EpisodeMemory`: train-only historical past keys, real completed history/future, provenance-checked residuals; per-query causal future-end cutoff.
3. `HierarchicalRetriever`: Region Top-64 → Node Top-16 with mask/support-conditioned scoring.
4. `PropagationTransport`: sparse local-road historical future + residual movement, time-shift distribution, valid forecast evidence.
5. `SetConditionalSelector`: two-type residual/evolution signed benefit, joint set context, Top-8 episode budget, node/horizon weights + learned Null.
6. `ForecastIntegrator`: shared hidden evidence token injection into frozen STLAformer (layers configured) plus explicit forecast correction.
7. `RiskController`: node/horizon predicted gain and learned scaling in `[0,1.25]` relative to exactly defined no-evidence baseline.
8. Train-only `CounterfactualTeacher`: leave-one-out with weight-to-Null, same-weight candidate replacement, and train-time horizon-block optimal action supervision.
9. `analysis`: 16 fixed-checkpoint inference interventions, legal random/shuffle controls, evidence interaction, block bootstrap, per-horizon/coverage/risk analysis, detailed files, and Markdown/HTML summaries.

### Scientific scope

Every module contribution is a **fixed-checkpoint intervention**. It does *not* claim what would occur if that module were removed and a new model retrained. Traffic-state probabilities are learned proxies unless sufficiently grounded road direction/distance and speed/density metadata exist. In undirected-only datasets, directional conclusions are disallowed.

## Project location

Unzip **at `Speed_LLM_RAG/`** so that `Speed_LLM_RAG/FARE`, `Speed_LLM_RAG/Backbone`, and `Speed_LLM_RAG/Base_SupportBench` are siblings. Set:

```bash
export SPEED_LLM_RAG_ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
```

### Preflight

```bash
python -u FARE/scripts/preflight.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target speed
python -m pytest -q FARE/tests
```

External `Base_SupportBench` resolved caches and official STLAformer best checkpoints **are not packaged**.

### Pilot: full FARE model

Pilot uses the complete architecture and all stages, with exploratory `checkpoint_in_sample` historical residuals. These are **not OOF**; do not treat Pilot residual-attribution claims as definitive scientific proof.

```bash
python -u FARE/scripts/run_pipeline.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target speed --seed 10

# Fast complete-pipeline smoke against real external data contracts
python -u FARE/scripts/run_pipeline.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target flow --seed 10 --smoke
```

#### Gadi — one job, one V100, all phases (default)

`submit_gadi.py` submits **one PBS job per target**, not two. In the same
`gpuvolta` allocation, `full_pipeline.pbs` executes:

1. Train the complete FARE (uses Validation loss each epoch for checkpoint selection).
2. Automatically evaluate **Test s20/s40/s60/s90 only** (default).
3. Automatically run fixed-checkpoint internal attribution on those Test splits,
   including Train-free LOO / Replacement / Scaling audit on Test s40.
4. Generate CSV/JSON/Markdown/HTML reports.

```bash
python -u FARE/scripts/submit_gadi.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target speed --seed 10 --checkpoint /abs/STLA_speed/best.pt
python -u FARE/scripts/submit_gadi.py --config FARE/configs/pilot.yaml \
  --dataset TraffiDent_Net_06 --target flow --seed 10 --checkpoint /abs/STLA_flow/best.pt
```

To also evaluate/analyze the Validation split **after** training, add
`--eval-val` to each submission. This **does not** affect validation-based
checkpoint selection during training, which is always performed.
The standalone `run_pipeline.py`, `evaluate.py`, and `analyze.py` also accept
`--eval-val` / `--test-only`.

Each submitted job requests **12 CPUs, one V100, 32 GB, and a 48h PBS walltime**.
The full internal attribution (16 modes × 4 test severities) can be expensive;
if it exceeds the remaining walltime the job is terminated, even though its
best checkpoint is saved. In that case a separate recovery invocation may be
needed; one-job scheduling is not a guarantee that everything fits in 48h.

**Existing previously submitted Train/Analysis jobs do not change** when code
is updated. Cancel/reconcile those old jobs before submitting a new unified
Pilot, and do not replace files while a running PBS process is using them.

### Formal: **requires auditable, chronologically OOS residuals**

The best existing STLAformer checkpoint may already have seen all Train labels. You cannot recover true OOS residuals from that checkpoint by merely separating the Memory indices. The formal configuration deliberately requires an `external_oos` file whose producer models were trained before the windows they predict.

Prepare an OOS fold manifest (see `docs/OOS_RESIDUALS.md`). The project can **generate predictions from existing fold checkpoints**; it does not silently train a new Backbone:

```bash
python -u FARE/scripts/build_oos_predictions.py \
  --manifest oos_folds.json --config FARE/configs/formal.yaml \
  --target speed --dataset TraffiDent_Net_06 \
  --output FARE_cache/oos_speed.npz

python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml \
  --target speed --dataset TraffiDent_Net_06 --seed 10 \
  --oos-forecast-file FARE_cache/oos_speed.npz
```

**Do not fabricate `trained_until` fields.** The fold manifest must be supported by actual dated training splits/checkpoint logs. If chronologically OOS pretrained checkpoints do not exist, formal will stop with a clear error; the explicitly marked exploratory Pilot remains runnable.

## Outputs

`FARE_exps/<target>/<dataset>/<pilot|formal>/<timestamp>_seed<seed>/`:

- `best.pt`, `last.pt`, `train_metrics.jsonl`, stage teacher cache + hash, manifests.
- `evaluation/performance_summary.csv`: Test s20/s40/s60/s90 by default, optionally Val with Avg(s20–s90), Avg(s20–s60), MAE/RMSE/MAPE and directly masked metrics.
- `evaluation/<severity>/full.json`, `.npz`: strict `experiment_common.metrics.metric_bundle` and raw prediction traces.
- `analysis/module_contributions.csv`: same checkpoint effects of each internal intervention including random/shuffled legal Memory.
- `analysis/evidence_interactions.csv`, `evidence_usage.csv`, `traffic_support_analysis.csv`, `risk_diagnostics.csv`, `horizon_metrics.csv`.
- `analysis/counterfactual/candidate_counterfactual.csv`: LOO, same-weight replacement, scaling on a query budget.
- `analysis/research_findings.json`, `research_report.md`, `research_report.html`.

### Standalone re-evaluation/analysis (no retraining)

```bash
python -u FARE/scripts/evaluate.py --checkpoint /abs/run/best.pt
python -u FARE/scripts/analyze.py --checkpoint /abs/run/best.pt
```

`analyze.py` is **read-only** on model weights; the checkpoint, Backbone and Memory hashes are validated.

## Known validation boundary

Local tests in this package exercise the true project data shapes with synthetic fixtures, mock Backbone and mock shared metric implementation. The official external STLAformer vendor source and genuine `Base_SupportBench` data caches are not included, so a formal Gadi run has **not yet been verified**. Run preflight + smoke on your Gadi checkout before submitting full Pilot and Formal. No numerical MAE/RMSE gains are claimed without actual experimental evaluation.

The model uses bounded sparse neighborhood transport and learnable lag alignment. Without reliable directed road metadata, it uses a documented undirected topological proxy; it must not be described as physically validated backward congestion-wave speed.
