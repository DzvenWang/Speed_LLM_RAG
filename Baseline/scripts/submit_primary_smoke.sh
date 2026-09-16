#!/bin/bash
set -euo pipefail
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
for m in ImputeFormer LRTC-TMCP GinAR RAFT RAST SARAF ReMFT; do
  echo "Submitting primary smoke $m"
  qsub -v METHOD="$m" pbs/methods/smoke_primary.pbs
done
