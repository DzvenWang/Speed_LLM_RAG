# Baseline v1.3.2 Minimal

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

## Runtime compatibility fixes (v1.3.5)

- ReMFT resolves its local BERT checkpoint in this order: `REMFT_BERT_PATH`, project-level `model_cache/bert-base-uncased`, then legacy `vendor/ReMFT/LLM/BERT`.
- ReMFT preflight and runner use the same resolver; no duplicate BERT copy is required.
- Smoke child processes force UTF-8 (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`) so vendor diagnostic output such as SARAF's sigma symbol is safe on Windows.
- No model architecture, hyperparameter, Protocol, ExperimentCache, or Frozen Backbone contract is changed.
