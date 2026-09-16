# Recent TRC Baselines in Baseline v1.0.0

The primary experiment now includes two recent TRC representatives with different roles, while MoGERNN/SPRINT remain implemented as secondary experiments because their native information contracts differ from the strict dynamic incomplete-H12 comparison.

## LRTC-TMCP — TRC 2024 — PRIMARY

- **Role:** recent optimization/tensor-completion reconstruction baseline.
- **Official source:** `vendor/LRTC-TMCP`, especially `lrtc-tmcp-demo.ipynb` and `utils.py`.
- **Main-table adaptation:** each TraffiDent query is completed independently using only the current incomplete H12. The tensor layout is `nodes × 1 query-day × H12`.
- **No extra evidence:** no forecast horizon, later observations, other validation/test queries, or extra historical days are visible to completion.
- **Observed protection:** observed H12 entries are restored exactly after completion.
- **Forecasting:** the completed H12 is passed to the same frozen PatchSTG reference used by ImputeFormer.
- **Determinism:** one Formal run (`seed10`) is sufficient; the seed is retained only for artifact naming/protocol consistency.
- **Safety gate:** run `tests/smoke_lrtc_tmcp_contract.py` locally and the Gadi GPU smoke before Formal. If the strict `N×1×H12` geometry makes the method degenerate, move it to a reconstruction-specific secondary table rather than relaxing the primary information budget.

Gadi commands:

```bash
qsub -v METHOD=LRTC-TMCP pbs/methods/smoke_primary.pbs
qsub -v METHOD=LRTC-TMCP,SEED=10 pbs/methods/formal_primary.pbs
```

## ReMFT — TRC 2026 — PRIMARY

- **Role:** LLM-enhanced traffic-forecasting comparator.
- **Official source:** architecture from `vendor/ReMFT/run.ipynb`.
- **Adaptation:** official model architecture is instantiated for benchmark `H=12, P=12`; stock data/train/test cells are not executed.
- **Selection:** Train is used for fitting, Val for checkpoint selection, and Test is only accessed after Formal selection.
- **Runtime:** requires the local BERT checkpoint under `vendor/ReMFT/LLM/BERT`.
- **Interpretation:** ReMFT is not claimed to be a native missing-sensing method; it tests whether a recent LLM traffic forecaster transfers to incomplete sensing without explicit predictive-support retrieval.

This clean release does not reuse previous ReMFT runs. ReMFT must pass the runtime contract and be rerun from Development before Formal evaluation.

## MoGERNN — TRC 2025 — SECONDARY

- **Role:** inductive forecasting for unobserved/sensor-free locations.
- **Reason for secondary status:** its native setting is structural partial sensing, whereas the primary benchmark uses sample-specific dynamic temporal/regional/directional incompleteness.
- **Implementation:** preserved in the project as a secondary method; previous experiment artifacts are not reused.
- **Current priority:** do not spend primary-experiment resources on it until the seven main Baselines are complete.

## SPRINT — TRC 2025 — SECONDARY

- **Role:** specialized reconstruction method using recurring historical-day context.
- **Reason for secondary status:** its native recurring-day context supplies additional causal evidence beyond a strict current-H12 input, so it should not silently enter the input-matched primary table.
- **Implementation:** preserved unchanged.

## Reporting rule

Primary Baseline table:

```text
ImputeFormer
LRTC-TMCP
GinAR
RAFT
RAST
SARAF
ReMFT
(+ PatchSTG backbone-only reference, + proposed method)
```

Secondary methods stay available for targeted robustness/generalization/appendix experiments and are not deleted from the repository.
