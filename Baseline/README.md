# Baseline v1.4.0

Experiment execution for TraffiDent_Net_06 under incomplete sensing.

## Primary methods
ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, ReMFT.

## Protocol
- Formal train/validation: s40, seed 10
- Evaluation: s20, s40, s60, s90
- Targets: speed and flow

## Explicit reference Backbone
There is **no `FROZEN_BACKBONE.json` and no Backbone freeze/registration step**.

Methods that reconstruct history before forecasting (`ImputeFormer`, `LRTC-TMCP`, and secondary `SPRINT`) receive an explicit Backbone model, for example:

```text
--backbone STLAformer
```

The resolver searches the corresponding Backbone Formal output:

```text
Backbone_exps/<target>/<backbone>/TraffiDent_Net_06/formal/
```

and uses the latest compatible seed-10 `best.pt` unless `--backbone-checkpoint` is supplied. The exact checkpoint path and SHA256 are stored in the Baseline Formal result. Multi-severity evaluation then reloads that exact checkpoint, so later Backbone retraining cannot change an existing Baseline experiment.

Backbone-dependent results are separated by model:

```text
Baseline_exps/<target>/ImputeFormer/<backbone>/TraffiDent_Net_06/...
Baseline_exps/<target>/LRTC-TMCP/<backbone>/TraffiDent_Net_06/...
```

## Local smoke
```powershell
python -u Baseline\scripts\smoke_primary_independent.py --continue-on-error
python -u Baseline\scripts\smoke_primary_reference_backbone.py --backbone STLAformer --continue-on-error
```

## Gadi examples
```bash
qsub -v METHOD=ImputeFormer,TARGET=speed,SEED=10,BACKBONE=STLAformer -N Impute_speed Baseline/pbs/methods/formal_primary.pbs
qsub -v METHOD=LRTC-TMCP,TARGET=speed,SEED=10,BACKBONE=STLAformer -N LRTC_speed Baseline/pbs/methods/formal_primary.pbs
qsub -v METHOD=GinAR,TARGET=speed,SEED=10 -N GinAR_speed Baseline/pbs/methods/formal_primary.pbs
```

For reconstruction methods, changing `BACKBONE=PatchSTG` or another supported Backbone creates a separate experiment rather than changing a global Backbone state.

### Matched STLAformer retrieval controls (v1.4.1)

`RAST-STLAformer` and `RAFT-STLAformer` preserve each method's native retrieval/fusion logic while replacing its direct forecasting backbone/path with the explicitly selected trained STLAformer checkpoint. No global freeze manifest is used. Training records the exact STLAformer checkpoint and SHA256, and multi-severity evaluation reloads that same checkpoint.

From v1.4.3, `RAFT-STLAformer` uses a minimal zero-initialized learnable residual coefficient, `final = STLAformer + alpha * (RAFT_fused - STLAformer)`, with `alpha=0` initially. This is a scale-compatibility adapter for transplanting RAFT fusion onto a strong pretrained backbone; RAFT retrieval/fusion itself is unchanged. `RAST-STLAformer` is architecturally unchanged and now prints the same base-vs-final helpful/harmful diagnostics during smoke and Formal training.

Gadi examples:

```bash
qsub -v METHOD=RAST-STLAformer,TARGET=speed,SEED=10,BACKBONE=STLAformer -N RAST_STLA_speed Baseline/pbs/methods/formal_primary.pbs
qsub -v METHOD=RAFT-STLAformer,TARGET=speed,SEED=10,BACKBONE=STLAformer -N RAFT_STLA_speed Baseline/pbs/methods/formal_primary.pbs
```

Use `TARGET=flow` for the corresponding flow experiments.


## v1.4.4 matched-backbone retrieval controls

Native `RAFT` and `RAST` are not modified. For matched-backbone controls only, both PatchSTG and STLAformer variants use a zero-initialized learnable residual coefficient so the initial prediction equals the selected reference Backbone. This is a compatibility calibration for transplanted retrieval/fusion branches, not a change to the official native baselines.

### Backbone reference training mode (v1.4.5)

Reference-Backbone-dependent Baselines may now select a Backbone trained on
`incomplete` (default, original `formal`) or `complete` (new `formal_complete`)
inputs via `BACKBONE_TRAIN_MODE=complete` or `--backbone-train-mode complete`.
Baseline's missing-query training and Core-5 evaluation do **not** change.
Complete reference runs and evaluation outputs are written to separate `/complete/`
directories; previous incomplete paths are untouched. See
[`REFERENCE_BACKBONE_TRAIN_MODES.md`](REFERENCE_BACKBONE_TRAIN_MODES.md) for commands,
validation rules and output paths.
