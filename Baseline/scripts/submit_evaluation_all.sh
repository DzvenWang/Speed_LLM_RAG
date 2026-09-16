#!/bin/bash
set -euo pipefail
# Primary Baseline evaluation only. LRTC-TMCP is deterministic and uses seed10 only.
for m in ImputeFormer GinAR RAFT RAST SARAF ReMFT; do
  for s in 10 20 30; do
    qsub -v METHOD="$m",SEED="$s" pbs/evaluation_main.pbs
  done
done
qsub -v METHOD="LRTC-TMCP",SEED=10 pbs/evaluation_main.pbs
