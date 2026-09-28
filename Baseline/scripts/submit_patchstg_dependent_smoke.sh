#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
cd "$ROOT"
for m in ImputeFormer LRTC-TMCP; do
  echo "Submitting $m primary PatchSTG-dependent GPU smoke"
  qsub -v METHOD="$m" pbs/methods/patchstg_dependent_smoke.pbs
done
