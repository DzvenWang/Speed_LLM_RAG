## v5.0.3 — Faster two-fold OOF with 4 independent V100 jobs
- `submit_oof.py` now defaults to batch_size=64, workers=4, and distinct `FARE_cache/oof_v503` producers.
- Forwards `OOF_WORKERS` through per-fold, finalize, and serial PBS scripts; replaces workers=0 throughput bottleneck.
- Persistent DataLoader workers/prefetch for train epochs, retained strict OOF temporal and provenance checks.
- Added per-epoch elapsed time, training time and peak allocated CUDA memory logging.
- Preserved PBS `afterok`: 4 producers -> finalizer with paired OOF and memory -> FARE Speed and Flow formal training/evaluate.
- New tests assert CLI and PBS argument wiring and that all producer jobs are independently submitted.

## v5.0.2 — Four independent V100 OOF producers + PBS automatic FARE handoff
- Time-frozen, read-only validated fold plans for each producer.
- `--fold-id` trains exactly one producer; `--finalize-only` requires all producers completed.
- PBS `afterok` dependencies ensure no memory construction or FARE training before verified OOF completion.
- Optional `--auto-fare` queues Speed/Flow formal train+auto-evaluate jobs after OOF.
- Atomic epoch checkpoints, completion markers with best checkpoint hash, independent v502 caches.

## v5.0.1 — Automated Chronological Producer OOF

- New `fare/oof.py`, `scripts/run_oof_pipeline.py`, `scripts/submit_oof.py`,
  `pbs/oof_pipeline.pbs`.
- Actual time-purged STLAformer fold training with fold-local normalization,
  paired incomplete/complete historical predictions and auditable producer checkpoints.
- Resume within fold at epoch granularity and across completed folds.
- Row-identity episode-keyed OOF with no `window_start` deduplication.
- Memory schema 6 preserves failure family/severity/fold/source row and rejects
  old timestamp-keyed OOF and old memory.
- Strict source and provenance verification in `preflight.py` and memory builder.
- No change to STLAformer vendor, Backbone official runner, or shared cache files.

# Changelog

## 5.0.0 — Retrieval identity and counterfactual utility redesign

- Intentionally incompatible full FARE project, reusing shared data, graph and
  Backbone interfaces but not old v4 Memory/Checkpoints/Teacher formats.
- Added paired chronological OOF Failure influence Memory schema 5.
- Added candidate/per-horizon counterfactual utility and hard-negative learning.
- Added explicit learnable forecast support deficit, gain, uncertainty, harm heads.
- Added set-level redundancy penalty and dynamic NULL selection with exact
  no-evidence fallback.
- Replaced temporal-mean evidence injection with ordered horizon-preserving
  convolutional tokens and evidence-mass control.
- Unified risk control over hidden and forecast prediction differences.
- Added matched-random and candidate-prior controls, new v5 training and tests.
- Formal default and early OOF/source preflight, retained V100 PBS and live logs.

Known deployment gate: chronological OOF producer STLA checkpoints, Base_SupportBench
caches, and STLAformer frozen checkpoint must be present externally.
