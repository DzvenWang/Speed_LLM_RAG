#!/bin/bash
set -euo pipefail
cat <<'TXT'
Baseline v1.1.0 Core-5 Primary-7 workflow

Train/Val: s40 x 5 Core families
Evaluate: s20/s40/s60/s90 x the same 5 Core families

1) qsub pbs/prepare_formal.pbs
2) bash scripts/submit_primary_smoke.sh
3) bash scripts/submit_primary_development.sh
4) bash scripts/submit_ginar_stability_probe.sh 10
5) bash scripts/submit_primary_formal_seed10.sh
   -> each successful Formal job automatically evaluates its exact best.pt on s20/s40/s60/s90
6) python scripts/primary_run_status.py

No Baseline FORMAL_SELECTION.json or FROZEN_BASELINE_PROTOCOL.json is required.
For a training-only debug job use AUTO_EVAL=0 explicitly.
TXT
