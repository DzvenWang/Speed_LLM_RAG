#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
STAGE=${STAGE:-development}
SEED=${SEED:-10}
python scripts/state_probe_preflight.py --stage "$STAGE"
qsub -v STAGE="$STAGE",SEED="$SEED" pbs/predictive_need_state_probe.pbs
