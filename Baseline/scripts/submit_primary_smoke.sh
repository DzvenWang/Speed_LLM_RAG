#!/bin/bash
set -euo pipefail
cd ${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
for m in ImputeFormer LRTC-TMCP GinAR RAFT RAST SARAF ReMFT; do
  echo "Submitting primary smoke $m"
  qsub -v METHOD="$m" pbs/methods/smoke_primary.pbs
done
