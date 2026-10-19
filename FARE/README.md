# FARE v5.0.3 — 4-V100 parallel OOF, batch 64, four DataLoader workers


Recommended Gadi workflow (Speed and Flow, 2 folds each, each fold an independent 1×V100/12-CPU PBS job):

```bash
python -u FARE/scripts/submit_oof.py --target both --nfolds 2 --warmup 0.30 --purge 12 --epochs 100 --batch-size 64 --workers 4 --parallel-folds --auto-fare --plan-only
python -u FARE/scripts/submit_oof.py --target both --nfolds 2 --warmup 0.30 --purge 12 --epochs 100 --batch-size 64 --workers 4 --parallel-folds --auto-fare
```

Four independent PBS producer jobs feed one dependent GPU finalizer, which validates and builds paired OOF; two formal FARE GPU jobs then train and automatically evaluate. Independent fold jobs can start on different nodes. Never submit the manual FARE commands if using `--auto-fare`. Details: [docs/FARE_V5_0_3_BATCH64_4WORKERS.md](docs/FARE_V5_0_3_BATCH64_4WORKERS.md).

---

## Previous release: v5.0.2

Recommended Gadi workflow:

```bash
python -u FARE/scripts/submit_oof.py --target both --nfolds 2 --warmup 0.30 --epochs 100 --batch-size 16 --parallel-folds --auto-fare --plan-only
python -u FARE/scripts/submit_oof.py --target both --nfolds 2 --warmup 0.30 --epochs 100 --batch-size 16 --parallel-folds --auto-fare
```

Full instructions: `FARE/docs/FARE_V5_0_2_FOUR_GPU_PIPELINE.md`. Each Fold requests 1 V100/12 CPUs. PBS dependencies gate finalization and both formal FARE train+auto-evaluate jobs. To submit FARE manually after OOF, omit `--auto-fare` and use `FARE/scripts/submit_gadi.py` (commands in the guide).

---

# FARE v5.0.1 (Utility-Aligned + Automated Chronological OOF)

**New in v5.0.1:** One-command chronological STLAformer fold training,
paired held-out forecast generation, verified failure-conditioned OOF Memory,
and duplicate-time-safe Episode identity. Details and Gadi commands:
[docs/FARE_V5_0_1_AUTOMATED_OOF.md](docs/FARE_V5_0_1_AUTOMATED_OOF.md).

---

# FARE v5.0.1 — Forecast-Utility Aligned Retrieval under Incomplete Sensing

**Full source project** for Net-01/02/03/04/06, Speed and Flow, based on the connected
`Base_SupportBench` and `Backbone` projects. **No FARE v4 checkpoint or Memory compatibility.**

## Research hypothesis

Historical evidence is useful not because the past resembles the current history,
but because the *completed historical episode* contains trustworthy information
which corrects the current frozen STLAformer's remaining forecast error under its
specific sensing failure. The model must learn to select and **reject** evidence.

## Implemented architecture

1. **Strong fixed STLAformer** (the supplied Backbone checkpoint): base forecast,
   reusable hidden Prefix, learnable evidence-bearing attention suffix.
2. **Forecast Support Deficit**: prediction-horizon-specific demand for *beneficial*
   evidence, trained from counterfactual candidate gains rather than mask ratio alone.
3. **Chronologically OOF paired Memory**: observed historical state, residual,
   history-vs-complete-availability forecast difference, historical future,
   valid masks, causal timestamps. A standalone producer script generates the
   train-only paired predictions using external chronological fold checkpoints.
4. **Coarse recall**: graph-region state/capability routes, sparse candidate pool
   with strict historical completion and purge eligibility checks.
5. **Forecast Utility Reranker**: per-candidate/per-horizon gain, calibrated
   uncertainty, and harmful-evidence probability. Multiple single-candidate
   counterfactual forwards supply train-only supervised labels; listwise,
   pairwise, Huber and harm losses all use these same labels.
6. **Hard-negative learning**: high-similarity, low-ranked and random shortlisted
   historical cases are evaluated for *real* negative gain; no pseudo-positive/
   negative labeling based solely on failure type or similarity.
7. **Set selection and NULL**: query-conditioned score plus redundancy penalty,
   optional no-evidence selection, and horizon-wise normalized candidate mass.
8. **Identity-preserving temporal injection**: Conv1d, temporal positional
   embedding, early/late order-sensitive features, historical Residual/Evolution/
   Failure streams, and weighted candidate tokens for the frozen STLAformer suffix.
9. **Unified bounded risk gate**: final prediction `base + gate*(evidence-base)`,
   where gate includes explicit non-null mass; no evidence means exactly base.

## Constraints & important scientific limitations

- **Training and selection use train labels only.** Val is for model selection,
  Test s20/s40/s60/s90 for *final assessment*. If Test is used for exploratory
  diagnostic intervention, label it as such and do not repeatedly tune on it.
- Paired OOF producer checkpoints must have been trained on an earlier temporal
  period and independently audited. A manifest's `trained_until` timestamp alone
  is a claim, not proof; independently check the producer training protocol.
- Both `prediction` and `prediction_complete` must be calculated using the **same
  OOF producer model**, with observed vs fully available historical training
  observations. The latter is a counterfactual input, not a claim of a separately
  trained complete-sensing backbone.
- The packaged system **does not train historical OOF STLAformer fold checkpoints**;
  their chronological training depends on the external `Backbone` project's
  data-split protocol. Do not use the old full-data STLAformer checkpoint to
  invent retrospective OOF predictions.
- Formal runs fail closed if v5 paired OOF predictions or Memory are missing.
- This release passed local CPU unit/integration tests with a synthetic data
  fixture; it **has not yet been run against real TraffiDent data on a Gadi V100**.
  Successful smoke/preflight on your real system is required before formal PBS.
- The v5 project is not compatible with v4 `best.pt`, old Memory directories or
  the old `diagnose_once` artifact protocol. Archive old results for comparison.

## Recommended layout at workspace root

```text
Speed_LLM_RAG/
├── FARE/                 # this project
├── Base_SupportBench/    # existing shared dataset/canonical metrics
├── Backbone/             # STLAformer source/checkpoints
├── FARE_exps/            # generated checkpoints (do not overwrite v4)
├── FARE_cache/           # NEW schema 5 Memory
└── FARE_logs/            # live logs
```

## Quick CPU checks

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:${PYTHONPATH:-}"
python -m pytest -q FARE/tests
```

## Formal OOF preparation

Prepare an externally trained chronological fold manifest:

```json
{
  "folds": [
    {
      "checkpoint": "/absolute/path/to/chronological_fold_01.pt",
      "trained_until": 25000,
      "evaluate_from": 25012,
      "evaluate_until": 40000
    },
    {
      "checkpoint": "/absolute/path/to/chronological_fold_02.pt",
      "trained_until": 39999,
      "evaluate_from": 40012,
      "evaluate_until": 60000
    }
  ]
}
```

Numbers **above are placeholders**, not Net-06 cutoff values. `trained_until`
means last historical training **target** timestamp, not last training sample's
start time. The intervals must be consistent with your cached Train windows.

```bash
python -u FARE/scripts/build_oos_predictions.py \
  --config FARE/configs/formal.yaml \
  --target speed --manifest /actual/oof_speed_folds.json \
  --output FARE_cache/oof_speed_paired.npz

python -u FARE/scripts/build_oos_predictions.py \
  --config FARE/configs/formal.yaml \
  --target flow --manifest /actual/oof_flow_folds.json \
  --output FARE_cache/oof_flow_paired.npz
```

### Preflight before queuing

```bash
python -u FARE/scripts/preflight.py --config FARE/configs/formal.yaml \
  --target speed --oos-forecast-file FARE_cache/oof_speed_paired.npz
python -u FARE/scripts/preflight.py --config FARE/configs/formal.yaml \
  --target flow --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

### Direct local development smoke

```bash
python -u FARE/scripts/run_pipeline.py --config FARE/configs/formal.yaml \
  --target speed --oos-forecast-file FARE_cache/oof_speed_paired.npz --smoke
```

### Single full train + automatic evaluation PBS per target

```bash
export FARE_PYTHON="$(command -v python)"
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml \
  --target speed --oos-forecast-file FARE_cache/oof_speed_paired.npz
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml \
  --target flow --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

The helper performs CPU checks before calling `qsub`. PBS uses `gpuvolta`,
**one V100, twelve CPU cores, 32 GB RAM, 48-hour limit**, timestamped live
logs, a stored resource monitor, full train and automatic speed/flow-specific
s20/s40/s60/s90 evaluation with MAE, RMSE and Avg(s20–s90).

Use `python -u FARE/scripts/evaluate.py --help` for post-training manual
re-evaluation. It reuses the saved FARE checkpoint, paired Memory, and
provenance-checked Backbone; it does not resume training.

## Retrieval-specific validation (after training, separately if desired)

```bash
python -u FARE/scripts/analyze.py --checkpoint /actual/v5/best.pt --eval-val
```

This runs the same fixed v5 model under Full, No Evidence, Random historical
Memory, Shuffled Evidence, **Matched Random shortlist**, Query-independent
candidate-prior and similarity-only modes. **Retrained** controls using matched
training budgets should also be conducted before claiming methodological
superiority. The fixed-checkpoint audit is descriptive, not equivalent to an
independently trained baseline.

See `docs/FARE_V5_MODEL_DESIGN.md` and `docs/FARE_V5_GADI.md`.
