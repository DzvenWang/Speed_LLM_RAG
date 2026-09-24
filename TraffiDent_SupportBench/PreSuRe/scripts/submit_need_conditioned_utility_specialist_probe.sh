#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
STAGE=${STAGE:-development}
SEED=${SEED:-10}
python scripts/need_conditioned_specialist_preflight.py --stage "$STAGE"
qsub -v STAGE="$STAGE",SEED="$SEED" pbs/need_conditioned_utility_specialist_probe.pbs
