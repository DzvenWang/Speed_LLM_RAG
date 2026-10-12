#!/usr/bin/env bash
# Submit 8 formal jobs: 2 retrieval methods x 2 STLA initializations x 2 targets.
# Run from the Speed_LLM_RAG repository root, after unpacking this archive.
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
cd "$ROOT"
SEED="${SEED:-10}"
DATASET="${DATASET:-TraffiDent_Net_06}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DRY_RUN="${DRY_RUN:-0}"
case "$SEED" in *[!0-9]*|'') echo "Invalid SEED=$SEED" >&2;exit 2;; esac
for mode in incomplete complete; do
  for target in speed flow; do
    if [[ "$mode" == incomplete ]]; then
      if [[ "$target" == speed ]]; then stamp=20260930_215233_seed10; else stamp=20260930_220624_seed10; fi
      stage=formal
    else
      stamp=20261009_170304_seed10
      stage=formal_complete
    fi
    checkpoint="$ROOT/Backbone_exps/$target/STLAformer/$DATASET/$stage/$stamp/best.pt"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 3; }
  done
done
for method in raft rast; do
  case "$method" in
    raft) project=Mask_RAFT-STLA; prefix=MRAFT23 ;;
    rast) project=Mask_RAST-STLA; prefix=MRAST23 ;;
  esac
  [[ -f "$project/pbs/formal.pbs" ]] || { echo "Missing $project/pbs/formal.pbs" >&2; exit 3; }
  for mode in incomplete complete; do
    if [[ "$mode" == incomplete ]]; then
      short=I; stage=formal
    else
      short=C; stage=formal_complete
    fi
    for target in speed flow; do
      if [[ "$mode" == incomplete ]]; then
        if [[ "$target" == speed ]]; then stamp=20260930_215233_seed10; else stamp=20260930_220624_seed10; fi
      else
        stamp=20261009_170304_seed10
      fi
      checkpoint="$ROOT/Backbone_exps/$target/STLAformer/$DATASET/$stage/$stamp/best.pt"
      name="${prefix}${short}_${target}"
      args="TARGET=$target,SEED=$SEED,DATASET=$DATASET,BATCH_SIZE=$BATCH_SIZE,BACKBONE_MODE=$mode,BACKBONE_CHECKPOINT=$checkpoint"
      printf 'SUBMIT %-18s MODE=%-10s TARGET=%-5s BACKBONE=%s\n' "$name" "$mode" "$target" "$checkpoint"
      if [[ "$DRY_RUN" == 1 ]]; then
        printf '  qsub -N %s -v %s %s/pbs/formal.pbs\n' "$name" "$args" "$project"
      else
        qsub -N "$name" -v "$args" "$project/pbs/formal.pbs"
      fi
    done
  done
done
