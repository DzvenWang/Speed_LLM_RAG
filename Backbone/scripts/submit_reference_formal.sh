#!/bin/bash
set -euo pipefail
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
MODEL=$(python - <<'PY'
import yaml
c=yaml.safe_load(open('configs/backbone_selection.yaml'))
print(c['selection']['reference_model'])
PY
)
mapfile -t SEEDS < <(python - <<'PY'
import yaml
c=yaml.safe_load(open('configs/backbone_selection.yaml'))
for x in c['experiment']['formal']['seeds']: print(x)
PY
)
for SEED in "${SEEDS[@]}"; do
  JID=$(qsub -v MODEL="$MODEL",STAGE=formal,SEED="$SEED" pbs/backbone_select.pbs)
  echo "$MODEL formal seed=$SEED -> $JID"
done
