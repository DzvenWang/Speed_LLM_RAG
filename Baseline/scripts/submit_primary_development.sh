#!/bin/bash
set -euo pipefail
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
for m in ImputeFormer LRTC-TMCP GinAR RAFT RAST SARAF ReMFT; do
  echo "Submitting primary development $m seed10"
  qsub -v METHOD="$m",SEED=10 pbs/methods/development_primary.pbs
done
