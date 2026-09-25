#!/bin/bash
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-/scratch/mh20/zw1165/Speed_LLM_RAG}"
cd "$ROOT/Backbone"

CHECKPOINT="${1:-${CHECKPOINT:-}}"
SEED="${2:-${SEED:-10}}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: bash submit_checkpoint_evaluations.sh <checkpoint> [seed]" >&2
  echo "Default: one combined GADI job for s20/s40/s60/s90 with one final summary table." >&2
  echo "Set PARALLEL=1 to submit four independent per-severity jobs." >&2
  exit 2
fi
if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="$ROOT/$CHECKPOINT"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 3
fi

if [[ "${PARALLEL:-0}" == "1" ]]; then
  for SEVERITY in 20 40 60 90; do
    echo "Submitting independent checkpoint evaluation: severity=$SEVERITY seed=$SEED"
    qsub -v "SEVERITY=$SEVERITY,SEED=$SEED,CHECKPOINT=$CHECKPOINT,SPEED_LLM_RAG_ROOT=$ROOT" pbs/evaluate_checkpoint_severity.pbs
  done
else
  echo "Submitting combined checkpoint evaluation: s20/s40/s60/s90 seed=$SEED"
  qsub -v "SEED=$SEED,CHECKPOINT=$CHECKPOINT,SPEED_LLM_RAG_ROOT=$ROOT" pbs/run_checkpoint_evaluations.pbs
fi
