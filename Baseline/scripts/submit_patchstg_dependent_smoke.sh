#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
cd "$ROOT"
for m in ImputeFormer LRTC-TMCP; do
  echo "Submitting $m primary PatchSTG-dependent GPU smoke"
  qsub -v METHOD="$m" pbs/methods/patchstg_dependent_smoke.pbs
done
