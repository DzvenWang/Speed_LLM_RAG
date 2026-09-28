#!/bin/bash
set -euo pipefail
cd ${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
for m in ImputeFormer LRTC-TMCP GinAR RAFT RAST SARAF ReMFT; do
  echo "Submitting primary development $m seed10"
  qsub -v METHOD="$m",SEED=10 pbs/methods/development_primary.pbs
done
