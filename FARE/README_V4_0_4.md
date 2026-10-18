# FARE v4.0.4 Complete Project

This archive includes the actual model/training/evaluation source plus a **native, single-job** comprehensive diagnostic. See [docs/FARE_V4_0_4_ONE_PASS.md](docs/FARE_V4_0_4_ONE_PASS.md) for installation, CPU preflight, and the one-qsub command.

- Main training entry: `scripts/run_pipeline.py` (unchanged workflow; v4.0.3 checkpoints remain readable).
- Original formal PBS: `pbs/full_pipeline.pbs`.
- New diagnostic GPU entry: `pbs/diagnose_once.pbs`.
- New local CPU preflight: `scripts/diagnose_preflight.py`.
- Full project source under `fare/`, tests under `tests/`.

**Data qualification:** the default four-severity diagnostic uses existing Test evaluation caches for exploratory audit only, clearly flagged in each output. Unbiased model selection requires held-out severity-specific validation data. No model retraining occurs in the diagnostic job.
