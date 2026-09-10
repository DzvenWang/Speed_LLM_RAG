#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
"$PY" "$ROOT/scripts/source_lock.py" verify --root "$ROOT" --sources "$ROOT/configs/backbone_selection.yaml"
find "$ROOT/vendor" -mindepth 2 -maxdepth 2 -name .git -type d -prune -print -exec rm -rf {} +
find "$ROOT/vendor" -mindepth 2 -maxdepth 2 -name .git -type f -print -delete
"$PY" "$ROOT/scripts/source_lock.py" verify --root "$ROOT" --sources "$ROOT/configs/backbone_selection.yaml"
echo "Vendor .git metadata removed; source-tree verification remains active."
