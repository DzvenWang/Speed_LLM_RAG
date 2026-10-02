# Backbone v1.1.10 Slim

## v1.1.10 Unified Formal lifecycle and path fix

- Direct/local `scripts/run_model_formal.py` now completes the full Formal lifecycle: training -> `best.pt` -> automatic s20/s40/s60/s90 checkpoint evaluation.
- Gadi `pbs/run_model_formal.pbs` uses the same lifecycle and no longer invokes a second duplicate post-training evaluation.
- `scripts/post_training_evaluate.py` resolves config/checkpoint paths before launching its child evaluator, fixing the local `Backbone\Backbone\configs\backbone.yaml` failure when a repository-relative config path is supplied.
- `scripts/submit_formal.sh` submits PatchSTG through the same unified Formal PBS entry point as the other five models.
- Smoke behavior remains bounded and unchanged: `smoke_all_formal.py` controls smoke evaluation separately.
- No model architecture, recipe, hyperparameter, Base_SupportBench protocol, ExperimentCache, metric, or explicit-only checkpoint-freeze semantics are changed.


Minimal Backbone project for TraffiDent incomplete-sensing experiments.

## v1.1.9 Gadi PBS environment fix

- Based directly on Backbone v1.1.8 (`configs/backbone.yaml`).
- `pbs/run_model_formal.pbs` is self-contained for the current Gadi TFP environment: `TFP_ENV_ROOT` may override the environment, otherwise it falls back to `/scratch/mh20/zw1165/envs/TFP`.
- Formal training still uses the environment's absolute `bin/python`; timestamp streaming uses absolute `bin/python3`.
- No model architecture, recipe, Formal protocol, ExperimentCache, training logic, or post-training evaluation logic is changed.

## Active models
STAEformer, HimNet, BigST, STDN, PatchSTG, STLAformer.

## Active contract
- targets: `speed`, `flow`
- Formal seed: `10`
- Formal training: s40 cache
- automatic post-training evaluation: s20 / s40 / s60 / s90
- training never freezes a checkpoint
- freeze is explicit via `scripts/freeze_checkpoint.py`

## Main entry points

### Local 12-case smoke
```powershell
python -u Backbone\scripts\smoke_all_formal.py
```

### Gadi Formal submission (recommended)
```bash
Backbone/scripts/submit_formal.sh STAEformer speed
Backbone/scripts/submit_formal.sh PatchSTG flow
```
Models: STAEformer, HimNet, BigST, STDN, PatchSTG, STLAformer. The submitter creates `Backbone_logs/<target>/<model>/` before `qsub`, routes the merged PBS stdout/stderr to a timestamped `.OU` file there, and the PBS job writes its `.live.log` to the same directory.

### Explicit freeze
```bash
python -u Backbone/scripts/freeze_checkpoint.py \
  --model PatchSTG --target speed --seed 10 --checkpoint /path/to/best.pt
```

## Runtime chain
`preflight -> train -> best.pt -> evaluate checkpoint at 20/40/60/90 -> NOT_FROZEN`

## Deliberately removed
Legacy screening/selection, automatic model selection, old frozen-test runners,
layout-recovery utilities, STDN diagnostic side-project, duplicate severity configs,
legacy cache migration helpers, old submission wrappers, and historical migration docs.

`vendor/` model sources are intentionally not refactored by this slim release.
