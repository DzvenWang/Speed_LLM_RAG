#!/bin/bash
set -euo pipefail
for m in GinAR ImputeFormer RAFT RAST SARAF; do for s in 10 20 30; do qsub -v METHOD="$m",SEED="$s" pbs/evaluation_main.pbs; done; done
