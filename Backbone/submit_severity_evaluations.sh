#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for s in 20 40 60 90; do
  prep=$(qsub -v SEVERITY="$s" pbs/prepare_severity_test.pbs)
  echo "s${s} prepare: $prep"
  evaljob=$(qsub -W depend=afterok:"$prep" -v SEVERITY="$s" pbs/evaluate_frozen_severity.pbs)
  echo "s${s} eval:    $evaljob"
done
