#!/bin/bash
set -euo pipefail
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
for m in ImputeFormer GinAR RAFT RAST SARAF ReMFT; do
  for s in 20 30; do
    echo "Submitting remaining primary formal $m seed$s"
    qsub -v METHOD="$m",SEED="$s" pbs/methods/formal_primary.pbs
  done
done
