# Migration: PreSuRe v0.6.0 -> v0.7.0

## Why this update exists

Development v0.6 established that the SoftMix target is not the bottleneck. On Net-06 Development, `DiagOracleSoftMix_M3Top4` retained roughly 51% Masked headroom, while the learned M4 recovered only about 0.2%. M4 also matched the teacher's mean support mass (~0.85) but remained close to PatchSTG, showing that the missing capability is candidate-specific query-support compatibility and correction estimation rather than intervention frequency.

## Scientific change

v0.7 replaces the compact v0.6 M4 student with **Pairwise Predictive-Support Compatibility Learning**.

M3 is kept as a shortlist generator. For every query node and M3 Top-K historical support, M4 now consumes:

- M1 node embedding for query and support;
- complete deployable query node context;
- complete historical support node context;
- sensing state for query and support;
- historical support residual trajectory across all forecast horizons;
- query and historical-support PatchSTG forecast trajectories;
- M3 relative shortlist confidence.

Separate query/support context encoders and a residual-trajectory encoder feed a pairwise interaction network. Candidate self-attention compares the Top-K supports before node-horizon utility and residual-transport heads are evaluated.

The validated v0.6 SoftMix teacher is retained. v0.7 additionally applies a horizon-local pairwise utility-ranking loss so the representation must preserve which support is better, not merely match average support mass.

## M5

The v0.6 continuous intervention teacher is retained. M5 is intentionally not redesigned until M4 produces materially stronger corrections.

## Compatibility

- Shared Experiment Contract is unchanged.
- PatchSTG checkpoint contract is unchanged.
- retrieval / SupportBench artifacts are unchanged.
- M1 and M3 checkpoint definitions are unchanged but the full pipeline reruns them.
- v0.6 M4 checkpoints are intentionally rejected; rerun M4 under v0.7.
- Development and Formal still use one PBS job and one V100.

## Validation metrics added for M4

The M4 summary now reports future-labelled Validation diagnostics for analysis only:

- teacher top-1 agreement;
- pairwise utility ranking accuracy;
- mixture-correction distillation loss;
- support-mass MAE.

Validation futures are not model inputs and are never used for deployment inference.
