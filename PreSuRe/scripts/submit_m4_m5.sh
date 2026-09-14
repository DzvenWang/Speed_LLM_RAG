#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
STAGE=${STAGE:-development}
ARGS=(--stage "$STAGE"); if [[ "${FORCE:-0}" == "1" ]]; then ARGS+=(--force); fi
"$PY" scripts/check_continuation.py "${ARGS[@]}"
jid(){ echo "$1" | sed 's/\.gadi-pbs$//'; }
Jm4=$(qsub -v VARIANT=M4,STAGE=$STAGE,SEED=10 pbs/variant.pbs); echo "M4 region-horizon routing -> $Jm4"
Jm5=$(qsub -W depend=afterok:$(jid "$Jm4") -v VARIANT=M5,STAGE=$STAGE,SEED=10 pbs/variant.pbs); echo "M5 local safe gate -> $Jm5"
