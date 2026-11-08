# TCRR-STLA v5.3.0 P0 Deep Mechanism Diagnostics (diagnostic-only release)

The model and checkpoint version remain **5.3.0**. This release adds a **read-only analysis pass**. It does not change architecture, loss, adapter, retrieval memory, formal TRAIN, FINALIZE or POSTPROCESS by default.

## Additions

- `tcrr_stla/v5/p0_deep_diagnostics.py`: streamed, same-checkpoint paired `M000 / M100 / M110 / M111` diagnostics: C1/C2/C3 MAE gains, C2 candidate utility, valid candidate ranking, best single candidate + None hindsight oracle, optional top-score retrieval baseline, initial/final mixture gains, support harmful-weight share, C3 location-level harm, normal/CF gates, relative harm AUC/Brier/ECE, conditional severity, horizon, observation-rate, correction-size, fault-family/direct-mask breakdowns, and hypothetical risk-coverage policies.
- `scripts/evaluate_v5.py`: opt-in `--p0-deep` path uses the **same** existing checkpoint / teacher / bank / retrieval / masks and the same batched inference path as `evaluate_v5.py`; it does not require `--diagnostics` and never alters inference decisions.
- `pbs/p0_deep_diagnostics_v5.pbs`: one PBS job does Validation selection and, optionally, four-severity test post-hoc diagnosis; defaults 1 V100 / 12 CPU / 32GB / 12h.
- `pbs/submit_p0_deep_v5.sh`: preflight checks finalized v5.3.0 Controller checkpoint and SHA/identity-matched S0 adapter before submitting.
- `scripts/p0_summarize_v5.py`: compact combined Speed/Flow report and contiguous query-index block-bootstrap intervals.
- `tests/test_v53_p0_deep.py`: reproducibility, masks, false positive, candidate availability, and paired mechanism metrics.

## Formal run commands (Gadi)

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/TCRR-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
python -m pytest -q TCRR-STLA/tests/test_v53_p0_deep.py

export SPEED_RUN="$PWD/TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v5/formal/20261028_155629_seed10_full"
export FLOW_RUN="$PWD/TCRR-STLA_exps/flow/incomplete/TraffiDent_Net_06/v5/formal/20261028_155629_seed10_full"

# Optional small smoke: validation first 2 batches, no TEST, Speed only.
P0_MAX_BATCHES=2 P0_RUN_TEST=0 bash TCRR-STLA/pbs/submit_p0_deep_v5.sh "$SPEED_RUN"

# Full evaluation: all selection queries and all s20/s40/s60/s90 test queries.
P0_MAX_BATCHES=0 P0_RUN_TEST=1 bash TCRR-STLA/pbs/submit_p0_deep_v5.sh "$SPEED_RUN"
P0_MAX_BATCHES=0 P0_RUN_TEST=1 bash TCRR-STLA/pbs/submit_p0_deep_v5.sh "$FLOW_RUN"
qstat -u "$USER"

# When both jobs finish successfully:
python TCRR-STLA/scripts/p0_summarize_v5.py --root "$PWD"
```

## Output paths

In each run directory:

- `p0_diagnostics/validation_selection.live.log`
- `p0_diagnostics/test.live.log`
- `evaluation/validation/automated/p0_deep_selection/correct__correct/validation/p0_deep/`
- `evaluation/test/automated/p0_deep_test_readonly/correct__correct/s{20,40,60,90}/p0_deep/`

Each `p0_deep/` includes:

- `summary.json`: canonical metrics + metadata and warnings
- `per_query.csv`: one row per query, not a huge tensor dump
- `grouped_c3.csv`: conditional severity-free strata (each severity has own folder)
- `hypothetical_risk_coverage.csv`: transparent risk-threshold counterfactual curves, **not deployed performance**
- `worst_c3_queries.csv`: 30 worst average-harm queries

Cross-target summary after completing both runs:

- `TCRR-STLA_exps/p0_diagnostics_summary/comparison.csv`
- `TCRR-STLA_exps/p0_diagnostics_summary/P0_readout.md`
- `TCRR-STLA_exps/p0_diagnostics_summary/existing_negative_controls.csv` and `existing_c3_variants.csv` (if old POST outputs exist; separate check-subset population)

## Causal/scientific boundaries

- All C1/C2/C3 effects are **shared-weight toggles on the same Controller checkpoint**. Speed's formally selected model is C12, so this diagnosis does **not** redefine Speed's final model.
- Candidate single oracle (with a None fallback) uses future truth only **after** prediction for hindsight. A learned mixture might beat the best single candidate: signed oracle-minus-mixture is not an upper bound on all possible mixture policies.
- The C3 risk head was trained for harm relative to Backbone, **not** C3-on versus C3-off. Both risk discriminations are computed, but not equivalent.
- Hypothetical threshold curves include a fallback-to-M110 comparator: this is an analysis tool, not the implemented C3 rejection rule. Use Selection/Calibration-only to develop new thresholds; **never use Test to fit or choose them**.
- Test results are *descriptive* and do not authorize model revisions selected on the test set. If subsequent improvements are chosen based on Test insights, use a new untouched time/network split for final inference.
- `P0_MAX_BATCHES` is first-N deterministic batches, not a random sample. Full runs require `P0_MAX_BATCHES=0`.
- Bootstrap intervals resample groups of contiguous dataset indices (default 24 queries), rather than individual pixels. Not a substitute for multi-seed or cross-network generalization.
- High-volume predictions are not saved by default.
