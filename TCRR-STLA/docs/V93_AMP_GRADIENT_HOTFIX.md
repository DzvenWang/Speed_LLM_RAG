# TCRR-STLA v9.3.0 — AMP gradient-overflow hotfix 1

## Verified Gadi symptoms (2026-11-11)

- Speed PREP `183460513` and Flow PREP `183460664`: **success**, complete teacher train/val caches.
- Speed TRAIN `183460514` and Flow TRAIN `183460665`: both stopped just after residual epoch 1 with `Non-finite gradient during residual` (`v93/training.py` line 99 before patch).
- Their first residual validation MAE improved over frozen STLA, but no complete training result is available.

## Root cause and scope

In original v9.3, the AMP step does `scaler.scale(loss).backward()`, `scaler.unscale_(optimizer)` and immediately raises when gradients are non-finite. That prevents PyTorch GradScaler from performing its designed recovery (`scaler.step()` skips the unsafe update, `scaler.update()` decreases the scale).

This hotfix lets GradScaler handle **transient** FP16 gradient overflow while reporting skipped batches and scale changes. Persistent failures (32 consecutive skipped updates), nonfinite losses and FP32 nonfinite gradients remain fatal. It does **not** replace the finite checks, change model architecture, Teacher data or original configuration. The original underlying source of the numeric overflow is not independently measured on Gadi; the two logs demonstrate only the nonfinite gradients and early exit.

## Changed files

- `tcrr_stla/v93/training.py`: `_safe_optimizer_step()`; logs `V93_AMP_OVERFLOW`; epoch fields `optimizer_steps`, `amp_skipped_steps`.
- `tests/test_v93_amp_gradient_recovery.py`: CPU PyTorch AMP GradScaler overflow/skip/recovery and FP32 nonfinite regression tests.
- `docs/V93_AMP_GRADIENT_HOTFIX.md`: this file.
- `MANIFEST_SHA256.txt`: regenerated source checksums.

## Gadi: replace project only; do NOT rerun PREP

Update `${SPEED_LLM_RAG_ROOT}/TCRR-STLA` using the patched full-source ZIP, preserving the other project roots and generated cache / run directories.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"

export STLA_SPEED="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_FLOW="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"

# Inspect real existing checkpoint folders, do not guess timestamps.
find "$PWD/TCRR-STLA_exps/speed" -path '*/v9_3/incremental/formal/*/last.pt' -print
find "$PWD/TCRR-STLA_exps/flow" -path '*/v9_3/incremental/formal/*/last.pt' -print
```

Assign the exact existing directories printed above to `SPEED_RUN` and `FLOW_RUN`, then verify their `last.pt` files exist. For an automatically discovered newest run use:

```bash
SPEED_RUN="$(find "$PWD/TCRR-STLA_exps/speed" -path '*/v9_3/incremental/formal/*/last.pt' -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)"
FLOW_RUN="$(find "$PWD/TCRR-STLA_exps/flow" -path '*/v9_3/incremental/formal/*/last.pt' -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -f "$SPEED_RUN/last.pt" && test -f "$FLOW_RUN/last.pt"
printf 'SPEED_RUN=%s\nFLOW_RUN=%s\n' "$SPEED_RUN" "$FLOW_RUN"

# Stage-1 checkpoint should contain next_epoch=1 and stage_index=0.
python - "$SPEED_RUN/last.pt" "$FLOW_RUN/last.pt" <<'PY'
import sys,torch
for filename in sys.argv[1:]:
    ck=torch.load(filename,map_location='cpu',weights_only=False)
    print(filename,'stage_index=',ck['stage_index'],'next_epoch=',ck['next_epoch'])
PY

qsub -N V93_SPD_FIX \
  -v TARGET=speed,SEED=10,V93_STAGE=train,BACKBONE_CHECKPOINT="$STLA_SPEED",V93_MODE=incremental,V93_GRAPH_MODE=none,V93_RUN_DIR="$SPEED_RUN",V93_RESUME="$SPEED_RUN/last.pt" \
  TCRR-STLA/pbs/formal_v93.pbs

qsub -N V93_FLW_FIX \
  -v TARGET=flow,SEED=10,V93_STAGE=train,BACKBONE_CHECKPOINT="$STLA_FLOW",V93_MODE=incremental,V93_GRAPH_MODE=none,V93_RUN_DIR="$FLOW_RUN",V93_RESUME="$FLOW_RUN/last.pt" \
  TCRR-STLA/pbs/formal_v93.pbs
```

`--resume` only works with the **same original run** and identical Teacher identity / model config. Trainer restores model/optimizer/scaler/RNG and picks up residual epoch 2. The code changes above do not alter the training config signature. The PREP adapter and Teacher files are reused without recreation.

Monitor `V93_AMP_OVERFLOW` and epoch `amp_skipped_steps`. A few isolated skips are tolerable; repeated skips or non-finite losses require additional investigation. The patched code has not been formally exercised on the Gadi V100 from this offline environment.
