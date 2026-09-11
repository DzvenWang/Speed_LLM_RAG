#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
cd "$ROOT"
SEED=$(python - <<'PY2'
import yaml
c=yaml.safe_load(open('configs/backbone_selection.yaml'))
print(c['experiment']['screening']['seed'])
PY2
)
for M in STAEformer HimNet BigST STDN PatchSTG; do
  jid=$(qsub -v MODEL="$M",STAGE=screening,SEED="$SEED" pbs/backbone_select.pbs)
  echo "$M screening seed=$SEED -> $jid"
done
