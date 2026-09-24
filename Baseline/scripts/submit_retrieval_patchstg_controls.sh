#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
cd "$ROOT"
SEED=${1:-10}
for METHOD in RAFT-PatchSTG SARAF-PatchSTG RAST-PatchSTG; do
  echo "Submitting matched-backbone control $METHOD seed=$SEED"
  qsub -v METHOD="$METHOD",SEED="$SEED" pbs/methods/formal_secondary_patchstg.pbs
done
