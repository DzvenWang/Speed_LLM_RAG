#!/bin/bash
set -euo pipefail
# Run from Backbone/ after audit_time_features.py PASS.
# v4 priority order follows the evidence already obtained:
#   - E1 batch=16 was worse, so do not spend GPU on E3/E4 first.
#   - E2 tests the previously broken patience override.
#   - E5 isolates WD contribution.
for EXP in e2_pat30 e5_wd0; do
  echo "Submitting $EXP"
  qsub -v EXP="$EXP",STAGE=screening,SEED=10 STDN_Special/pbs/run_single.pbs
done
