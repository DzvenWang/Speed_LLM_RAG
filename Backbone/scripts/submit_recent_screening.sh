#!/bin/bash
# Stage-1B only: PatchSTG is the sole recent-model extension; do NOT resubmit the four already-completed v0.3.1 models.
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
cd "$ROOT"
/scratch/mh20/zw1165/envs/TFP/bin/python scripts/check_stage1a_ready.py
/scratch/mh20/zw1165/envs/TFP/bin/python scripts/validate_cache.py --stage screening
SEED=$(python - <<'PY2'
import yaml
c=yaml.safe_load(open('configs/backbone_selection.yaml'))
print(c['experiment']['screening']['seed'])
PY2
)
for M in PatchSTG; do
  jid=$(qsub -v MODEL="$M",STAGE=screening,SEED="$SEED" pbs/backbone_select.pbs)
  echo "$M recent-screening seed=$SEED -> $jid"
done
