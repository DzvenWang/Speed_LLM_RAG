#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
STAGE=${STAGE:-development}
SEED=${SEED:-10}
export STAGE SEED
echo "Submitting PreSuRe v0.9A.1 Safe Top-Selection: stage=$STAGE seed=$SEED"
qsub -v STAGE="$STAGE",SEED="$SEED" pbs/predictive_utility_safe_selection_probe.pbs
