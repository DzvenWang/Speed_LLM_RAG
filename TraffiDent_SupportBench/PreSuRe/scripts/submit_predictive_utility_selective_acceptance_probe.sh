#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
STAGE=${STAGE:-development}
SEED=${SEED:-10}
export STAGE SEED
echo "Submitting PreSuRe v0.9A.3 Selective Utility Acceptance: stage=$STAGE seed=$SEED"
qsub -v STAGE="$STAGE",SEED="$SEED" pbs/predictive_utility_selective_acceptance_probe.pbs
