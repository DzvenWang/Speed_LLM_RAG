**FARE v6 initialization policy:** FARE trainable modules always start from fresh seeded weights. Reuse only the frozen STLAformer Backbone checkpoint and paired chronological OOF outputs. See `docs/FARE_V6_FRESH_INIT_POLICY.md`.

# FARE v6.0.0 — Failure-Aware Residual Evidence Learning

Full experiment code for strong-backbone traffic forecasting under incomplete sensing, targeting Transportation Research Part C.

Three integrated research hypotheses:

1. **C1** Failure-Aware Predictive Support Modeling: paired OOF sensing-induced residual decomposition and recoverability-aware support-only predictor.
2. **C2** Counterfactual Utility-Guided Residual Retrieval: train-only strict single-candidate gains, direct ranking gradients to coarse retrieval projections, hard-negative comparison, and robust numeric training.
3. **C3** Horizon-Adaptive Risk-Controlled Evidence Transfer: temporally ordered OOF residual, per-horizon evidence mass, latent STLAformer injection and harm-aware NULL / Risk Gate.

Every full training runs **validation-only innovation audit** prior to final Test evaluation. Improvements in MAE alone cannot mark a research contribution as established: independent retraining, matched parameter budgets, multiple seeds and datasets are required.

## Deployment and exact Gadi PBS commands

Read **`docs/FARE_V6_FULL_IMPLEMENTATION.md`** before replacing a running version.

- Use existing `Backbone`, `Base_SupportBench`, schema-6 paired OOF `FARE_cache` and frozen STLAformer checkpoints.
- Run `python -m pytest -q FARE/tests` and the CPU `submit_gadi.py --check-only` preflight on Gadi before submitting.
- v5 FARE checkpoints **must never** initialize v6. All trainable FARE modules begin from fresh seeded weights. A v6 `--resume` is permitted only for explicitly resuming that same v6 experiment after interruption.
- Do not rerun completed OOF Producers.

## Engineering validation

Full local test suite includes original deployment/OOF tests, plus new strict counterfactual, coarse-recall gradient, numerical stability, support-only fallback and validation innovation-audit tests. Actual V100 memory headroom and real-data accuracy require Gadi verification.

## FARE v6.2 — Experiment 1 (separate, fresh-start decision pipeline)

See [`docs/FARE_V62_EXPERIMENT1.md`](docs/FARE_V62_EXPERIMENT1.md). New
`FARE/scripts/experiment1.py` and `submit_experiment1.py` implement full fresh
Support-only foundation, immutable historically legal CandidateBank, untrained
Direct Residual feasibility, T0/T1/T2/T3 independent transfer training, and
validation-only paired assessment. This is not v6.2 full joint C1/C2/C3 training.
