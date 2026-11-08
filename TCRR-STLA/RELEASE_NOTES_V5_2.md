# TCRR-STLA v5.2.0 — Protocol-corrected formal experiment release candidate

The research modules retain **CFRM (C1)**, **SCRT (C2)** and **CRSI (C3)**, including M000–M111 and automatic validation diagnostics. This update repairs numerical protocol flaws, control targets, model selection, and PBS postprocessing without reading any earlier TCRR outputs. It still requires a **real Gadi GPU smoke and end-to-end training** before claims of formal experimental success.

## P0 — Correct paired counterfactual calculation

- Normal and CF must use identical **original valid source masks**; the CF mask is their intersection with the current query mask.
- v5 now passes `strict_source_mask=True` for all Replay-Bank and dynamic paths.
- Counterfactual Replay Manifest schema 27 includes `replay_protocol=source_mask_intersection_v2`, the SHA256 of source observed/artificial missing masks and the frozen Teacher fingerprint. Existing incompatible Replay banks are invalidated automatically.
- Wrong-mask diagnosis uses *real* alternative query masks from a deterministic, same-section pool, with matched fault family/coverage preference. The sample set is identical across controls, and the result records changed-mask rates and coverage differences. A non-distinct negative control cannot silently report an advantage.
- CF pair ranking now learns both beneficial and harmful orderings based on actual training-label utility; no assumption forces the matched response to win.

## CRSI — Joint risk, shared structural scope, and constrained refinement

- A learnable joint-harm classifier sees the actual proposed **combined** Normal/CF correction (before rejection). Its BCE target is whether the final combined prediction harms the Backbone forecast, rather than a weighted average of individually harmful components.
- Correction control has truly shared coarse logits inside deterministic groups built from **train-only functional neighbours** and within horizon blocks; a bounded local deviation permits node/horizon refinement. Groups are not claimed to be physical upstream/downstream road segments.
- Explicit temporal and functional-neighbour gate smoothing and local-deviation regularizers.
- Streaming approximate joint-harm AUC and harm/coverage tradeoff CSV per diagnosis mode; automatic severity tables (`component_interaction_by_severity.csv` and `c3_gain_by_fault_family.csv`); predictions NPZ remain opt-in.

## Chronological selection/calibration/check

- Validation queries are partitioned by original `window_start` into selection (60%), calibration (20%) and locked check (20%), with purged input/forecast-window borders.
- S0, C12/C3 epoch selection and stage choice use only selection queries.
- Rejection thresholds use calibration queries only and are never enabled/disabled by looking at the locked check.
- The locked check is reported once after checkpoint selection; automatic mechanistic comparisons evaluate the same locked-check queries but do not change the trained weights or rejection thresholds.
- These are empirical, time-ordered checks, **not** distribution-free risk guarantees.
- Smoke intentionally reuses its tiny validation sample across roles and is not scientific performance evaluation.

## S0 and runtime

- S0 adapter identity now includes learning rate, epochs, batch size, data fingerprint, and mask protocol. Incompatible cached S0 is regenerated.
- The unused extra frozen Query Teacher forward is disabled for v5.
- All new checkpoints use `version=5.2.0`. Earlier 5.1.0 checkpoints are deliberately incompatible; **retrain from S0**.

## Automatic Gadi PBS sequence

Use `pbs/submit_pipeline_v5.sh` once for each target. PBS dependencies will run:

1. **TRAIN**, `formal_v5.pbs`, 48 hours, S0/Replay/C12/CRSI without postprocessing;
2. **DIAGNOSTICS**, `post_diagnostics_v5.pbs`, 24 hours, chronological-check factorial/negative/C3-variant diagnosis;
3. **TEST**, `post_test_v5.pbs`, 24 hours, all severity tests with the already-fixed checkpoint.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export SPEED_LLM_RAG_ROOT="$PWD" TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP
export TARGET=speed
export BACKBONE_CHECKPOINT="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
bash TCRR-STLA/pbs/submit_pipeline_v5.sh
```

Repeat with `TARGET=flow` and the Flow Backbone checkpoint. The submission prints all three PBS IDs and the run directory. A failed stage prevents the next stage from starting. To run only local GPU smoke before this, submit `pbs/formal_v5.pbs` with `SMOKE=1`.

**Study limitation:** source code validation and synthetic tests do not establish Gadi installation compatibility, walltime sufficiency, or improvement over STLAformer. Confirm all via genuine GPU execution and hold out the test data from architecture selection.
