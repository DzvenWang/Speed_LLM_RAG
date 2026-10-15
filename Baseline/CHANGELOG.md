## v1.4.8 — legacy Formal evaluation + source fingerprint warning (2026-10-15)

- Accept completed Baseline v1.3.10 Formal runs as eligible inputs to the
  existing direct Evaluate-only pipeline; also retain v1.4.0–v1.4.7.
- Official vendor source *tree SHA256* mismatch now emits
  `WARNING SOURCE_TREE_HASH_MISMATCH` and continues verification. The recorded
  `SOURCE_LOCK.json` fingerprint is **not** modified and a warning remains
  visible in PBS live logs. This applies to `source_lock.py verify` and
  `verify-one` (including preflight); lock creation still enforces original
  source state, and missing sources / lock metadata, remote, commit and
  entrypoints still fail as before.
- Keep strict Protocol/cache identity, completed-run status, method, seed,
  reference Backbone checkpoint SHA256, best.pt load and all evaluation
  protocol/metric validations. Old weights and experiment outputs are not
  rewritten by this patch.
- **Caution:** a warning-only source mismatch does not establish numerical
  reproducibility. Compare old MAE/RMSE after re-evaluation; document the
  changed source if using results in a paper.

# Baseline v1.4.4 — matched-backbone safe residual consistency

- Native `RAFT` and native `RAST` are unchanged and remain source-faithful primary baselines.
- `RAFT-PatchSTG` now uses the same zero-initialized learnable residual scale as `RAFT-STLAformer`.
- `RAST-PatchSTG` and `RAST-STLAformer` now use `base + alpha * (rast_fused - base)` with `alpha=0` initialization.
- RAST matched-control smoke/formal logs persist residual scale plus base-vs-final utility diagnostics.
- Direct evaluation now supports `RAFT-PatchSTG` and `RAST-PatchSTG` with exact reference-backbone provenance.
- Formal PBS supports both PatchSTG and STLAformer matched controls without changing native RAFT/RAST.

# Changelog

## v1.4.3
- `RAFT-STLAformer` now uses a minimal zero-initialized learnable residual scale: `final = base + alpha * (raft_fused - base)`, with `alpha=0` at initialization. This prevents an uncalibrated RAFT correction from immediately overwriting a strong pretrained STLAformer forecast while preserving native RAFT retrieval/fusion.
- The trained RAFT-STLAformer residual scale is persisted in Formal checkpoints and restored during s20/s40/s60/s90 evaluation. Native RAFT weights remain available under the legacy `model` checkpoint key for compatibility.
- `RAST-STLAformer` architecture is unchanged. Its smoke/epoch logs now expose the already-computed matched-backbone impact diagnostics: base/final normalized MAE, helpful/harmful/neutral fractions, correction magnitude, and mean absolute-error change.
- Bumped matched-control implementation revision tags for reproducibility.

## 1.4.1
- Added `RAST-STLAformer` matched-backbone retrieval control.
- Added `RAFT-STLAformer` matched-backbone retrieval control.
- Generalized retrieval-control bridges from PatchSTG-only to explicit reference Backbones while retaining PatchSTG compatibility aliases.
- Added exact STLAformer checkpoint/SHA provenance to training and evaluation.
- Added direct s20/s40/s60/s90 evaluation support for both new controls.
- Kept Baseline v1.4.0 Formal artifacts evaluation-compatible.
## v1.4.2
- Reference-Backbone and Baseline latest-run paths are project-root-relative and portable between local Windows and Gadi.
- Legacy absolute pointer/checkpoint paths are automatically rebased using `Backbone_exps` / `Baseline_exps` anchors when copied across machines.
- Formal summaries/checkpoints now persist the portable reference checkpoint path plus SHA256, rather than a host-specific absolute path.


## 1.4.5 — selectable Backbone training sensing mode
- Added `BACKBONE_TRAIN_MODE=incomplete|complete` (default `incomplete`) and `--backbone-train-mode` for all reference Backbone consumers.
- Complete checkpoints resolve from Backbone `formal_complete`; legacy incomplete checkpoints from `formal` remain unchanged.
- Strict verification against the checkpoint's `train_data_mode` (old checkpoints default `incomplete`).
- Mode-specific Baseline formal, evaluation, and live log paths prevent overwriting earlier experiments.
- Persist `reference_backbone_train_mode` with reference checkpoint provenance and keep old Formal results compatible.
- Added seven CPU-only regression tests; see `REFERENCE_BACKBONE_TRAIN_MODES.md`.

## v1.4.7 (MAPE evaluation + ReMFT lifecycle smoke helper fix)

- Uses the shared Base_SupportBench evaluator to output MAPE (%) for `overall`,
  `masked`, H3/H6/H12 and the five sensing-failure families.
- Includes `MAPE_count` and `MAPE_coverage` in severity/scenario summaries to
  disclose denominator exclusions for near-zero true targets.
- Retains historical run identity/metric revision, pretrained reference
  Backbone selection mode, MAE-based checkpoint selection and live logs.
- Carries forward v1.4.6's cleanup of invalid `save_store()` calls in ReMFT,
  GinAR, RAFT, and SARAF lifecycle-smoke helpers; RAST store persistence is kept.
