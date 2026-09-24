#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
STAGE=${STAGE:-development}
SEED=${SEED:-10}
python scripts/need_gated_probe_preflight.py --stage "$STAGE"
qsub -v STAGE="$STAGE",SEED="$SEED" pbs/need_gated_utility_probe.pbs
