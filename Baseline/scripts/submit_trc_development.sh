#!/bin/bash
set -euo pipefail
cd ${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
for m in MoGERNN SPRINT; do
  echo "Submitting secondary TRC development $m seed10"
  qsub -v METHOD="$m",SEED=10 pbs/methods/trc_development.pbs
done
