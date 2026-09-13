#!/bin/bash
set -euo pipefail
for m in GinAR ImputeFormer RAFT RAST SARAF; do for s in 10 20 30; do echo "Submitting $m seed$s"; qsub -v METHOD="$m",SEED="$s" pbs/methods/formal_main.pbs; done; done
