#!/bin/bash
set -euo pipefail
for m in GinAR ImputeFormer RAFT RAST SARAF; do echo "Submitting $m formal seed10"; qsub -v METHOD="$m",SEED=10 pbs/methods/formal_main.pbs; done
