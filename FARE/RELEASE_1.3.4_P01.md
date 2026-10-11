# FARE v1.3.4 — P0.1 Single-GPU diagnostic

Complete backward-compatible FARE v1.3.3 distribution + one-job P0.1 (Speed then Flow).
All original train/evaluate/ablate/PBS interfaces remain intact. No checkpoint
format or Memory NPZ schema change. The `FARE` inference forward adds only an
opt-in `return_diagnostics=False` switch; Retrieval adds `diagnostics=False`.

## Gadi single job

From `/scratch/mh20/zw1165/Speed_LLM_RAG` with `FARE/` replaced by this package:

```bash
python -u FARE/scripts/diagnose_p01.py --help
qsub -N FARE_P01 -v SEED=10,PILOT_STAMP=20261009_140229_322315_seed10 FARE/pbs/p01_diagnostic.pbs
qstat -u "$USER"
```

**Resources:** `gpuvolta`, 1 V100 GPU, 12 CPUs, 32GB RAM, 48h.
**Targets:** Speed and Flow **sequentially in the same PBS job**.
**Data:** Train-only saved `memory.npz`, frozen checkpoint `candidate_best.pt`,
Val s40 cache (5,000 rows) remasked by the *canonical* Base_SupportBench
`sensing.materialize_mask` for s20/s40/s60/s90. Must pass a 32-row exact
s40 mask parity audit before starting. Does **not** read Test labels.

**Outputs:**
`FARE_exps/diagnostics_p01/TraffiDent_Net_06/<timestamp>_<jobid>/`
with `speed/sXX/` and `flow/sXX/` containing `diagnostic_summary.json`,
`integrity_sensitivity.csv`, `alpha_sweep.csv`, `family_metrics.csv`,
`candidate_probes.csv`, `candidate_oracle_regions.csv`, `candidate_utility.json`.
Combined `combined_report.json`, `combined_report.md`, `combined_metrics.csv`.
Live log: `FARE_logs/diagnostics_p01/<dataset>/p01.<jobid>.<stamp>.live.log`.

### Scope and limitations

- P0.1 is diagnostic re-masking of **Validation** windows, not the official
  frozen paired evaluation `Test` protocol, and no Validation oracle goes into
  the retriever. Any failure of s40 parity causes immediate failure.
- Selected/Random Top-64/Random Global/Time-matched use the same Memory store,
  coherent historical X/valid/Latent/TOD/DOW for each ID, 24-step overlap check,
  and 3 seeds for random probes.
- Full validation prediction and Gate/alpha analysis covers every Severity.
  Inference sensitivity samples `P01_SENSITIVITY_BATCHES` batches (8 by default);
  Counterfactual candidate utility probes only `P01_PROBE_ROWS` rows (64 by
  default), 4 regions, up to 8 slots, to control V100 costs.
- `candidate_oracle_regions.csv` is a **privileged local-region intervention
  upper bound** based on validation truth. It is not an achievable Forecast MAE.
- `oracle_pointwise_mae` likewise uses validation truth. Best fixed alpha is
  selected from chronologically earlier Validation windows and separately
  measured on later Validation windows.
- Old v1.3.3 `no_memory` remains a special intervention returning supported
  FARE-STLA without a Memory; its `gate=0` does **not** signify standard Base
  fusion. P0.1 does not use this ambiguous mode.
- Fixed-checkpoint interventions do not replace retrained no-memory controls.

### Optional affordable smoke (same full project)

To check just first 2 batches and 8 candidate probe windows, submit one PBS
job with `-v P01_MAX_BATCHES=2,P01_PROBE_ROWS=8,P01_SENSITIVITY_BATCHES=1`.
This only smoke-tests diagnostics, not a full P0.1 result.

Run CPU-only unit tests locally (requires Torch, PyYAML, NumPy, pytest):
`python -m pytest -q FARE/tests/test_p01.py`
