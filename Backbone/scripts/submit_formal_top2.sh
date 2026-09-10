#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
EXP=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_exps
cd "$ROOT"
[[ -f "$EXP/screening_shortlist.json" ]] || { echo "Missing $EXP/screening_shortlist.json" >&2; exit 2; }
mapfile -t MODELS < <(python - <<'PY'
import json
x=json.load(open('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_exps/screening_shortlist.json'))
for m in x['shortlist']: print(m)
PY
)
mapfile -t SEEDS < <(python - <<'PY'
import yaml
c=yaml.safe_load(open('configs/backbone_selection.yaml'))
for s in c['experiment']['formal']['seeds']: print(s)
PY
)
for M in "${MODELS[@]}"; do
  for S in "${SEEDS[@]}"; do
    jid=$(qsub -v MODEL="$M",STAGE=formal,SEED="$S" pbs/backbone_select.pbs)
    echo "$M formal seed=$S -> $jid"
  done
done
