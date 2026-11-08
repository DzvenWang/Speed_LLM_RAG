
## TCRR-STLA v9.1.0 — revised CF implementation (2026-11-08)

The current independently versioned v9 implementation is **9.1.0**. Frozen Normal anchor, horizon-local CF-aware Utility, matched-candidate two-mask training, source-valid utility targets, reliability-weighted Functional/optional Physical graph context and reproducible cache identities are in `tcrr_stla/v9/`. Run `scripts/train_v9.py` via `pbs/formal_v9.pbs`. See [`docs/V9_1_IMPLEMENTATION.md`](docs/V9_1_IMPLEMENTATION.md) and [`docs/V9_RUNBOOK.md`](docs/V9_RUNBOOK.md). Old v9.0.0 checkpoints/replay are incompatible; legacy v5.3 source and root `VERSION` remain unchanged. Real Gadi training is not claimed.

# TCRR-STLA v9.0.0 — Counterfactual Utility-Guided Residual Transfer

**New in this archive:** `tcrr_stla/v9/` is a separate, complete implementation of the Normal-Anchored + Counterfactual-Utility model, paired-mask Teacher data preparation, signed Functional/optional physical graphs, TRAIN-only full-observation supervision, reusable replay banks, multi-stage training, checkpoint resume, validation/Test scripts and Gadi PBS launcher. Legacy v3–v5.3 code is preserved for comparison. **The v9 training code is separate from legacy v5.3 weights.**

Read [`docs/V9_RUNBOOK.md`](docs/V9_RUNBOOK.md) for commands and [`docs/V9_ARCHITECTURE.md`](docs/V9_ARCHITECTURE.md) for equations/data contracts. The main entry points are `scripts/preflight_v9.py`, `scripts/train_v9.py`, `scripts/evaluate_v9.py`, `scripts/diagnose_v9.py`, and `pbs/formal_v9.pbs`. New hyperparameters are defined by `configs/tcrr_stla_v9.yaml`.

**Version compatibility:** The repository-root `VERSION` remains `5.3.0` so the bundled legacy v5.3 regression scripts retain their original version contract. The new method is explicitly versioned `9.0.0` in `tcrr_stla/v9/VERSION`, `configs/tcrr_stla_v9.yaml`, all v9 checkpoint metadata and v9 run identities.

**Verification:** New CPU synthetic tests under `tests/v9/` cover preparation/replay, graph features, Normal/CF decomposition, Utility gradients, multi-stage training, and resume. The source archive does not contain your real Backbone model, Base_SupportBench cache or v7 physical graph artifacts; these remain external requirements for a full Gadi formal run. Do not interpret synthetic test success as measured forecasting performance.

---

**Current hotfix: TCRR v5.3.0 AMP BCE correction** — see `RELEASE_NOTES_V5_3_AMP_HOTFIX.md`. Cache and checkpoint version remain `5.3.0`.

**Latest release: TCRR v5.3.0 (resource optimisation)** — see `RELEASE_NOTES_V5_3.md`.

# TCRR-STLA v5.1.0 — CFRM / SCRT / CRSI

**Active research modules**

- **C1 — Counterfactual Failure Response Modeling (CFRM)**: learns the query-conditioned sensing-failure response of historical forecasting residuals, with normal/CF branches, temporal encoding and incremental predictive-utility supervision.
- **C2 — Support-conditioned Residual Transfer (SCRT)**: evaluates overlap and intervention exposure, combines historical corrections using support-aware ranking and mixture-aware utility, and retains Normal and CF corrections separately under the final common candidate weights.
- **C3 — Counterfactual Response-guided Selective Intervention (CRSI)**: uses normal/CF directional consistency, functional-neighborhood and forecast-horizon localization, and separate Normal/CF intervention gates. The full bounded C2 proposal is reconstructed exactly when both gates equal one.

All three modules have independent **C1/C2/C3 switches** (`M000`–`M111`). With `C3=OFF`, both component strengths equal one and no C3 parameter or calibration controls the final prediction. `M00`/`M10`/`M01`/`M11` remain fixed-gate C1/C2 comparators, distinct from three-switch modes.

## What changed from v5.0.0?

The implementation now preserves the two transferred correction components after final SCRT ranking and before the nonlinear bound. Symmetric allocation of the bounded correction makes the sum of the two components **exactly equal** to the original full proposal. CRSI then uses two independent gates, directional response consistency, cross-component conflict, functional-neighborhood averaging and horizon-block adjustment. Training includes component utility, component harm and interaction targets. Calibration uses the **actual** dual-gated correction, not an incorrect approximation based on `gate × proposal`.

The earlier five-scale correction controller is retained as a **trained comparison** (`c3_variant=scale`). Other variants are `component`, `agreement`, `localized`, `full`, with a rotating auxiliary forecast loss during C3 training. Same-checkpoint variant comparisons are mechanism diagnostics; definitive independent-architecture claims require separately trained capacity/budget-matched variants.

CFRM and SCRT retain the corrected full implementations of v5.0.0. Their historical retrieval, teacher, adapter and replay artifacts are **newly built under `v5` paths**, not read from deleted earlier TCRR outputs. Shared STLAformer Backbone and Base_SupportBench are unchanged.

For formulas, tensor dimensions, exact call chain, losses, checkpoint selection, tests and limitations see [`docs/V5_1_ARCHITECTURE_IMPLEMENTATION.md`](docs/V5_1_ARCHITECTURE_IMPLEMENTATION.md).

## Active entrypoints

| Entry | Function |
|---|---|
| `configs/tcrr_stla_v5.yaml` | Version 5.1.0, project configuration and CRSI hyperparameters |
| `scripts/train_v5.py` | Fresh S0/retrieval/teacher/replay → CFRM+SCRT → CRSI → automatic analysis and test |
| `scripts/evaluate_v5.py` | Validation/test, eight switches, matched negative controls and optional CRSI variant |
| `scripts/auto_diagnose_v5.py` | Auto four-switch, eight-switch, negative controls and C3 variant comparisons |
| `pbs/formal_v5.pbs` | Gadi gpuvolta formal pipeline, logging and auto post-training diagnosis |
| `tcrr_stla/v5/` | Active PyTorch architecture and trainer |

Former-source utilities used by the active data/retrieval pipeline remain in the repository, but no old TCRR experiment outputs or model checkpoints are required.

## Run on Gadi (v5.3.0)

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/TCRR-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
export TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP
export SPEED_CKPT="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export FLOW_CKPT="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
python -m pytest -q TCRR-STLA/tests

# Optional realistic-size memory stress test, isolated from formal caches:
TARGET=speed BACKBONE_MODE=incomplete BACKBONE_CHECKPOINT="$SPEED_CKPT" \
  BATCH_SIZE=64 SMOKE=1 SMOKE_PROFILE_FULL_BATCH=1 \
  bash TCRR-STLA/pbs/submit_pipeline_v5.sh

# Formal: one command submits TRAIN -> (DIAGNOSTICS + TEST) automatically:
TARGET=speed BACKBONE_MODE=incomplete BACKBONE_CHECKPOINT="$SPEED_CKPT" \
  bash TCRR-STLA/pbs/submit_pipeline_v5.sh
TARGET=flow BACKBONE_MODE=incomplete BACKBONE_CHECKPOINT="$FLOW_CKPT" \
  bash TCRR-STLA/pbs/submit_pipeline_v5.sh
```

Other launcher overrides: `TCRR_TRAIN_WORKERS`, `TCRR_EVAL_WORKERS`,
`DIAGNOSTIC_BATCH_SIZE`, `DIAGNOSTIC_MAX_BATCHES`, and
`DIAGNOSTIC_PRIMARY_MAX_BATCHES`. See
`RELEASE_NOTES_V5_3_PBS_BATCH_HOTFIX.md` for exact semantics.

Memory profiling with `SMOKE_PROFILE_FULL_BATCH=1` repeats the 24 smoke query
indices to construct a true-size train microbatch; it measures activation
memory but **not** statistically representative throughput or forecasting
accuracy. Standard smoke uses batch 3. Run the smoke job on a Gadi GPU allocation,
not the login node.

Active artifacts: `TCRR-STLA_exps/<target>/<mode>/<dataset>/v5/`,
`TCRR-STLA_cache/<target>/<mode>/<dataset>/v5/`,
`TCRR-STLA_logs/<target>/<mode>/v5/`.

**Verification status:** local synthetic tests and source syntax checks are
included. The current patch has not yet been exercised on a real Gadi V100.

---

## Latest extension: v9.3.0 Full-Observation Multi-Failure CF

See [docs/V93_FULL_SOURCE_IMPLEMENTATION.md](docs/V93_FULL_SOURCE_IMPLEMENTATION.md) and `configs/tcrr_stla_v93.yaml`.

* Complete **v9.2 source and regression tests are preserved**.
* Direct CF student uses no historical retrieval at inference.
* TRAIN-only full-reference masks and multi-family/severity masks produce residual / response / damage targets.
* Frozen Normal residual plus CF Utility / Null Gate, pairwise mask response and utility comparisons.
* Road graph implementation is self-contained (audited import from supplied v7.1 source). No extra v7 project install is required.
* CPU synthetic end-to-end integration and old v9/v9.2 tests are included; real Gadi experiments remain to be run.
