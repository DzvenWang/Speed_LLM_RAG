#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
cd "$ROOT"

METHODS=("$@")
if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(RAST SARAF ReMFT)
fi
for METHOD in "${METHODS[@]}"; do
  echo "Submitting evaluation recovery ${METHOD} seed10"
  qsub -v METHOD="$METHOD",SEED=10 pbs/evaluation_main.pbs
done
