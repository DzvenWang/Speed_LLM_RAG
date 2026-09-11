#!/bin/bash
set -euo pipefail
# Run from Backbone/.
# The existing official_control (MAE=2.231304 in the supplied log) is not resubmitted.

for EXP in d1_bs16 d2_bs16_pat30 d3_regularized d4_low_lr; do
  echo "Submitting $EXP"
  qsub -v EXP="$EXP",STAGE=screening,SEED=10 STDN_Special/pbs/run_single.pbs
done
