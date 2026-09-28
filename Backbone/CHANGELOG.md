# v1.0.10 maintenance — Oracle ownership cleanup

- Removed Oracle-specific prediction export/validation/PBS files from `Backbone`.
- Oracle prediction-contract preparation now belongs to `Oracle_Diagnostic`.
- Backbone remains responsible only for model training/evaluation, frozen-checkpoint management, adapters, and source-lock validation.
- No model, checkpoint, cache, Protocol, metric, or training contract changed; project version remains 1.0.10.

# v1.0.10 — Formal train → automatic Core-5 evaluation

- GADI `pbs/run_patchstg.pbs` now runs s20/s40/s60/s90 checkpoint evaluation immediately after a successful formal training run.
- The exact `best.pt` from the current run stamp is resolved and verified before evaluation starts.
- Evaluation output is streamed into the same PBS live log and ends with the severity/scenario summary tables.
- `AUTO_EVAL=1` is the formal default; use `AUTO_EVAL=0` to train without final evaluation. Development remains train/val only.
- `run_checkpoint_evaluations.py --stdout-only` supports parent-managed logging without creating a duplicate nested live log.

# v1.0.9 — Core-5 sensing protocol

- Primary family set is entry_random, sensor_outage, temporal_outage, regional_outage, directional_outage.
- Predictive regional/directional outages are excluded from primary Train/Val/Test.
- Shared contract requirement updated to ExperimentCommon 1.2.0 / Protocol 2.1.0 / Core-5 paired group.
- Checkpoint evaluation now fails clearly if a severity result does not contain exactly the configured five scenarios.

# Backbone v1.0.7 — GADI ExperimentCommon API compatibility

- Fixes formal/development training failure on GADI when an older ExperimentCommon is installed.
- Training cache validation now uses the stable `validate_cache(cfg, stage)` signature only.
- `allow_locked_evaluation_validation` is passed only when the installed ExperimentCommon actually supports it.
- Legacy ExperimentCommon evaluation caches remain usable under the Simple Evaluation policy without reintroducing the Protocol access lock.
- No PBS resource change; live logs and per-line timestamps remain unchanged.

# Backbone v1.0.5

- Added exact recovery for historical PatchSTG spatial layouts instead of bypassing layout mismatches.
- Historical checkpoint evaluation automatically uses an embedded layout or `spatial_layout.npz` sidecar when available.
- Added `scripts/recover_patchstg_layout.py` to scan historical SupportBench copies and accept only an exact fingerprint match.
- Added `pbs/recover_patchstg_layout.pbs` with timestamped live logs for Gadi recovery.
- Added diagnostic-only `--allow-current-layout-mismatch`; such results are explicitly marked non-comparable.
- New PatchSTG training now persists the complete spatial layout inside `best.pt` and as `spatial_layout.npz`.

# Backbone v1.0.4

- Fixes shared-cache identity normalization: `validate_shared_cache()` now validates via ExperimentCommon and returns Backbone's normalized run identity instead of the raw cache identity.
- Prevents false `Cache stage=None != evaluation` failures for valid multi-severity evaluation caches.
- Keeps strict stage/version/hash/fingerprint/contract validation; this is not a compatibility bypass.
- Evaluation remains unlocked from the legacy Protocol grant-file gate.
- Preflight now reports the effective unlocked evaluation policy instead of the obsolete grant-file state.

# v1.0.3 - unlocked evaluation and local live logging

- Removed the Protocol evaluation-access grant requirement from Backbone evaluation consumers.
- Frozen/hash/contract/cache-version validation remains mandatory.
- Historical checkpoint evaluation now works without `EVALUATION_ACCESS_GRANTED.json`.
- `run_checkpoint_evaluations.py` now creates a timestamped combined live log under `Backbone_logs/` on Windows and Linux.
- GADI per-severity live logs remain enabled.

# Changelog

## v1.0.1

- Fixed `pbs/run_patchstg.pbs` so direct Gadi submission no longer requires `SPEED_LLM_RAG_ROOT` to be exported.
- Uses the TFP environment Python executable directly instead of `conda activate` in the PBS shell.
- Added one timestamped live log per PatchSTG PBS run under `Backbone_logs/`.
- Added default `STAGE=formal` and `SEED=10`, while keeping `qsub -v` overrides.
- Suppresses the duplicate Python-side log during PBS runs; local runs keep the existing log behavior.

## v1.0.0

- Rebuilt Backbone around PatchSTG as the fixed numerical reference.
- Bound training to Protocol v2.0.1 `protocol_s40` development/formal.
- Bound caches to ExperimentCommon shared contract 1.1.0 / cache schema 2.
- Enforced 12k balanced development and 50k balanced formal training contracts.
- Added paired evaluation support for s20/s40/s60/s90.
- Added evaluation access guard: materialized Test caches cannot be consumed before Protocol authorization.
- Added three-seed formal freeze and three-seed multi-severity evaluation summaries.
- Removed hard-coded Windows/Gadi paths and uses `SPEED_LLM_RAG_ROOT` or project-root inference.
- Reduced active vendor surface to the pinned official PatchSTG repository.

## 1.0.2 — 2026-09-25
- Added cross-platform explicit PatchSTG checkpoint evaluation for historical or ad-hoc `best.pt` files.
- Added `scripts/run_checkpoint_evaluations.py` to run s20/s40/s60/s90 locally from one command.
- Added GADI `pbs/evaluate_checkpoint_severity.pbs` with timestamped live logs in `Backbone_logs/`.
- `submit_checkpoint_evaluations.sh` now requires an explicit checkpoint, preventing accidental reuse of the historical model after a new formal model is available.
- Explicit checkpoint results are written under `checkpoint_evaluation/` and never modify `FROZEN_BACKBONE.json`.

## 1.0.8 — Scenario-aware evaluation reporting

- Evaluation now exports explicit scenario/family metrics for the five missing-sensing families.
- Each evaluation run writes `test_scenario_metrics.csv` in addition to per-query metrics and JSON summary.
- Multi-severity checkpoint evaluation prints two compact final tables: severity summary and scenario summary.
- Combined checkpoint evaluation writes severity/scenario summary CSV files under `Backbone_logs`.
- Standard frozen evaluation summaries now report the same scenario metrics and aggregate them across formal seeds.
- Added `pbs/run_checkpoint_evaluations.pbs` for one GADI job covering s20/s40/s60/s90 with one combined live log and final tables.
- `submit_checkpoint_evaluations.sh` now defaults to the combined GADI job; set `PARALLEL=1` for four independent jobs.

## Oracle prediction export utility (2026-09-26)
- Added `scripts/export_oracle_predictions.py` for Core-5 s20/s40/s60/s90 evaluation prediction contracts plus Train-only memory predictions.
- Added strict export validator and a timestamped Gadi PBS wrapper.
- Export follows the same current-cache/current-PatchSTG-layout policy as `evaluate_checkpoint.py`.
- No model, training, Protocol, cache, metric, or checkpoint contract was changed; project version remains 1.0.10.

### 1.0.10 Oracle export source-lock compatibility fix
- Fixed `scripts/verify_official_sources.py` to use the current `source_lock.py verify --config ...` CLI.
- Updated `scripts/setup_official_models.sh` and `scripts/strip_vendor_git.sh` to remove obsolete `verify-one`, `--root`, and `--sources` arguments.
- Preserved the current PatchSTG-only immutable source lock contract and all training/evaluation/checkpoint semantics.
- No model, metric, Protocol, ExperimentCommon cache, or frozen checkpoint definition was changed.
