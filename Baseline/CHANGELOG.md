## 2026-09-26 — Primary seed10 failure-recovery hotfix

- GinAR Formal validation divergence after a finite best checkpoint is now a guarded early-stop; the previously selected Val-best checkpoint is preserved for evaluation. Stability-probe mode remains fail-fast.
- RAST direct evaluation now unwraps the `(prediction, impact)` return contract correctly.
- SARAF no longer overwrites the canonical `time_feature_source` identity field; its runner-specific source is recorded as `saraf_time_mark_source`.
- Direct evaluation validates `RUN_MANIFEST.json` as the authoritative Protocol/cache identity when present, allowing already-trained SARAF v1.1.0 runs with the legacy summary-key collision to be reused safely.
- ReMFT direct evaluation now supplies the required `use_amp` argument and uses the Formal evaluation batch size.
- `save_val_outputs` now rejects runner metadata that attempts to overwrite canonical run-manifest identity fields.
- Added `scripts/submit_evaluation_recovery_seed10.sh` and `tests/smoke_primary_failure_recovery_20260926.py`.

# v1.1.0 hotfix — Direct post-Formal multi-severity evaluation

- Primary Formal PBS jobs now evaluate the exact completed Formal run immediately after training.
- Evaluation order is s20 -> s40 -> s60 -> s90, followed by severity and Core-5 scenario summaries.
- Removed the active dependency on `FORMAL_SELECTION.json` and `FROZEN_BASELINE_PROTOCOL.json` for Baseline evaluation.
- `run_evaluation.py` accepts `--formal-run-dir` and validates the completed run against the current Formal cache identity, method, seed, status, validation finiteness and checkpoint SHA256.
- Frozen PatchSTG remains required for PatchSTG-dependent baselines; only the Baseline-level freeze/selection barrier was removed.
- `AUTO_EVAL=1` is the Formal PBS default; set `AUTO_EVAL=0` only for debugging/training-only runs.
- No change to Core-5 Protocol, cache identity, model hyperparameters, or Baseline experiment version.

# v1.1.0 hotfix — Frozen PatchSTG bridge compatibility

- Fixed `baseline/backbones/patchstg_bridge.py` calling the Baseline-only `verify-one` action on Backbone v1.0.10 `scripts/source_lock.py`.
- Frozen PatchSTG source verification now uses the Backbone-owned native API: `source_lock.py verify --config configs/backbone.yaml`.
- The bridge now reads the live Backbone configuration from `configs/backbone.yaml` instead of stale `backbone_selection.yaml`.
- No change to Core-5 Protocol, ExperimentCommon cache identity, baseline recipes, seeds, or evaluation policy. Existing Core-5 caches remain valid.

# v1.1.0 — Core-5 s40 training + paired multi-severity evaluation

- Migrated Train/Val to `protocol_core5_s40`.
- Main families are `entry_random`, `sensor_outage`, `temporal_outage`, `regional_outage`, `directional_outage`.
- Migrated to ExperimentCommon v1.2.0 / Protocol v2.1.0 / cache schema 2.
- Bound PatchSTG-dependent baselines to Backbone v1.0.10 Core-5 frozen reference.
- Evaluation now consumes paired s20/s40/s60/s90 caches and prints final severity/scenario tables.
- Old Baseline v1.0.0 results cannot enter v1.1.0 Formal selection.

# v1.0.0 — Clean Primary-7 Baseline

- Locked the Primary Baseline suite to ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF and ReMFT.
- Preserved SPRINT, SLPF, MoGERNN, LEAF, LEAF-Qwen1.5B and RAST-PatchSTG as secondary/controlled methods.
- Removed old-result version compatibility, migration and method-revision gates.
- Replaced multiple overlapping Primary PBS entry points with one clean smoke/development/formal workflow.
- Formal selection now accepts only current-version, current-identity, `selection_valid=true` results.
- GinAR divergence is now a hard Formal failure; added a full-Formal-population stability probe and more conservative Formal optimization settings.
- ReMFT now aligns frozen BERT runtime depth to the local pretrained checkpoint and hard-fails any remaining architecture mismatch.
- Retained strict causal H12 LRTC-TMCP + frozen PatchSTG main-table adaptation.
