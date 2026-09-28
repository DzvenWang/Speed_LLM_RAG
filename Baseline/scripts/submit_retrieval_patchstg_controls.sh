#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
cd "$ROOT"
SEED=${1:-10}
for METHOD in RAFT-PatchSTG SARAF-PatchSTG RAST-PatchSTG; do
  echo "Submitting matched-backbone control $METHOD seed=$SEED"
  qsub -v METHOD="$METHOD",SEED="$SEED" pbs/methods/formal_secondary_patchstg.pbs
done
