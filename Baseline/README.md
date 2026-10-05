# Baseline v1.3.10 Minimal

Minimal experiment-execution release for TraffiDent_Net_06 under incomplete sensing.
Model architectures, adapters, runners, hyperparameters, and the frozen experiment protocol are unchanged from v1.2.3. Development-only tests, diagnostics, setup helpers, status tools, and redundant PBS entrypoints were removed.

## Primary methods
ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, ReMFT.

## Protocol
- Formal train/validation: s40, seed 10.
- Evaluation: s20, s40, s60, s90.
- Targets: speed and flow.
- ImputeFormer and LRTC-TMCP require target-specific Frozen Backbone manifests (the validated PatchSTG bridge remains unchanged when selected_model=PatchSTG).

## Local smoke
Independent methods (GinAR, RAFT, RAST, SARAF, ReMFT):

```bash
python -u scripts/smoke_primary_independent.py --continue-on-error
```

Frozen-Backbone-dependent methods (ImputeFormer, LRTC-TMCP):

```bash
python -u scripts/smoke_primary_frozen_backbone.py --continue-on-error
```

All primary methods:

```bash
python -u scripts/smoke_all_primary.py --continue-on-error
```

## Gadi Formal
From the repository root:

```bash
export SPEED_LLM_RAG_ROOT=$PWD
bash Baseline/pbs/submit_primary_formal.sh
```

Each successful Formal run automatically evaluates s20/s40/s60/s90.


## Generic Frozen Backbone contract (v1.3.2)
The Baseline contract no longer requires `selected_model=PatchSTG`. `FROZEN_BACKBONE.json` chooses the Backbone model. The existing `FrozenPatchSTG` bridge is deliberately preserved unchanged for the experimentally validated ImputeFormer/LRTC-TMCP + PatchSTG path. Other selected Backbone models must provide an explicit validated consumer bridge before use; no PatchSTG tensor/normalization assumptions are silently reused.


## Timestamped smoke logs (v1.3.3)
Every line emitted by the primary smoke driver and every captured child-process stdout/stderr line is prefixed with `[YYYY-MM-DD HH:MM:SS]`. This applies to both independent and Frozen-Backbone-dependent aggregate smoke logs.

## Runtime compatibility fixes (v1.3.6)

- ReMFT resolves its local BERT checkpoint in this order: `REMFT_BERT_PATH`, project-level `model_cache/bert-base-uncased`, then legacy `vendor/ReMFT/LLM/BERT`.
- ReMFT preflight and runner use the same resolver; no duplicate BERT copy is required.
- Smoke child processes force UTF-8 (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`) so vendor diagnostic output such as SARAF's sigma symbol is safe on Windows.
- No model architecture, hyperparameter, Protocol, ExperimentCache, or Frozen Backbone contract is changed.

## Gadi PBS environment fix (v1.3.7)

- Based directly on Baseline v1.3.6 (`configs/baseline.yaml`).
- `pbs/methods/formal_primary.pbs` no longer falls back to the deprecated bare `python` command.
- `TFP_ENV_ROOT` may override the Python environment; otherwise Gadi uses `/scratch/mh20/zw1165/envs/TFP`.
- The PBS job exports a deterministic `SPEED_LLM_RAG_ROOT`, TFP `PATH`, and a single Base_SupportBench entry in `PYTHONPATH`.
- No Baseline model architecture, adapter, runner, hyperparameter, Formal/evaluation protocol, ExperimentCache, or Frozen Backbone contract is changed.



## Official-source bootstrap and method-scoped preflight (v1.3.9)

- Fixes Formal jobs stopping before training when `Baseline/vendor` or `SOURCE_LOCK.json` is absent.
- `scripts/preflight.py --method <METHOD>` now verifies only the official repository used by that method; unrelated secondary repositories no longer block a primary Formal run.
- `scripts/setup_official_source.py --name <METHOD>` is a one-time explicit bootstrap. Exact-policy sources are detached at the configured commit; resolve-once sources resolve the configured branch once and record the resulting full commit and tree SHA in `vendor/SOURCE_LOCK.json`.
- PBS jobs remain offline/reproducible consumers: they verify the frozen source lock and never clone or update repositories automatically.
- No model architecture, adapter, hyperparameter, Formal/evaluation protocol, ExperimentCache, or Frozen Backbone contract is changed.
## Frozen Backbone compatibility and version-record synchronization (v1.3.10)
- `project_version` in `FROZEN_BACKBONE.json` is provenance metadata, not a hard compatibility gate.
- Frozen Backbone compatibility continues to validate model identity, target, formal seed/checkpoint, SHA256, run identity, shared contract, cache schema, protocol, metrics, prediction contract, and time-feature identity.
- Project/runtime version records are synchronized to v1.3.10, including `VERSION`, `configs/baseline.yaml`, preflight output (via config version), and Primary Formal PBS lifecycle labels.
- No model architecture, hyperparameter, dataset split, protocol, cache, or metric definition is changed.



## LRTC-TMCP / ReMFT information-contract correction (2026-10-05)

- **LRTC-TMCP** keeps its multi-day location × day × within-day tensor. Its 28 complete support days are now selected **only from the frozen chronological Train archive**, at the same within-day slot and strictly before the query. Validation/test hidden ground truth is never used as external support. The current H12 slice remains the exact incomplete benchmark query and observed cells are immutable.
- **ReMFT** now retains the official notebook's native **H96 → P12** working scale with `patch_len=16` and `stride=8`. The final H12 is reused exactly from the shared benchmark cache. The extra causal H84 prefix is re-materialized under the same sensing view family/severity and exposes only observed cells, so native context is restored without granting complete held-out states.
- No other Baseline implementation or hyperparameter block is changed by this correction.
