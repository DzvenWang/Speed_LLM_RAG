#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
STAGE=${STAGE:-development}
SEED=${SEED:-10}
"$PY" scripts/preflight.py --stage "$STAGE"
J=$(qsub -v STAGE="$STAGE",SEED="$SEED",VARIANT_START=M4 pbs/variant_suite.pbs)
echo "M4->M5 VariantSuite (one V100) -> $J"
