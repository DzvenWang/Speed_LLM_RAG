#!/bin/bash
set -euo pipefail
# Core-5 multi-severity evaluation: each PBS job runs s20/s40/s60/s90. LRTC-TMCP is deterministic and uses seed10 only.
for m in ImputeFormer GinAR RAFT RAST SARAF ReMFT; do
  for s in 10 20 30; do
    qsub -v METHOD="$m",SEED="$s" pbs/evaluation_main.pbs
  done
done
qsub -v METHOD="LRTC-TMCP",SEED=10 pbs/evaluation_main.pbs
