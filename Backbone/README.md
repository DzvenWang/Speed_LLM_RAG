# Backbone v1.1.1 Slim

Minimal Backbone project for TraffiDent incomplete-sensing experiments.

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
