# FARE v6.0.0

- Explicit Support-only, Base and Full forecast paths for attribution.
- Lazy paired OOF decomposition and sensing-specific controls.
- Differentiable train-only coarse Candidate Ranking through actual retrieval scorer.
- Rotating, failure/severity-aware Counterfactual Teacher query sampling.
- Strict fixed-single candidate injection without learned NULL attenuation.
- Robust FP32 gain and uncertainty losses with factorized metrics.
- Horizon-specific candidate mass and explicit Risk input from Harm/Sigma.
- Stage-specific trainability and automatic fixed-checkpoint validation innovation audit.
- GPU-capable cached episode tensors + persistent DataLoader processes.
- CPU method-level tests and independent audit launcher.

See docs/FARE_V6_FULL_IMPLEMENTATION.md for restrictions and Gadi launch.

## Audit-only patch v6.0.1 (checkpoint ABI remains v6.0.0)
- Expanded stratified Validation-only attribution and fixed-alpha/NULL candidate oracle.
- Added same-shortlist policies using query-independent historical prior.
- Added horizon-independent shared-weight intervention and failure/horizon reports.
- Added read-only PBS audit submission from existing v6.0.0 best.pt and provenance.
- No training/model weights, existing OOF or Test data modified by the audit.

## v6.2.0 Experiment 1 decision pilot

- New `transfer_v62.py`: direct vs legacy vs source-relative historical evolution vs candidate-aware spatial/time alignment.
- New `candidate_bank_v62.py`: immutable train/validation candidate IDs with causal checks, sample order/hash provenance.
- New `experiment1_v62.py`: random-from-scratch Support foundation, read-only paired OOF, direct headroom probe, isolated Transfer training and paired bootstrap decision.
- New `scripts/{experiment1,submit_experiment1}.py` and `pbs/experiment1_v62.pbs`.
- Transfer corrections now verified to be applied to Support-only rather than raw STLA. No-candidate yields exact Support-only prediction.
- All claimed innovation statuses remain `INCONCLUSIVE` until required independent controlled tests.
