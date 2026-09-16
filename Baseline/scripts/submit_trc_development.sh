#!/bin/bash
set -euo pipefail
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
for m in MoGERNN SPRINT; do
  echo "Submitting secondary TRC development $m seed10"
  qsub -v METHOD="$m",SEED=10 pbs/methods/trc_development.pbs
done
