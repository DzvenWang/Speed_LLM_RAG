#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Backbone
PY=${PYTHON:-python}
"$PY" "$ROOT/scripts/source_lock.py" verify --strict-hash --config "$ROOT/configs/backbone.yaml"
find "$ROOT/vendor" -mindepth 2 -maxdepth 2 -name .git -type d -prune -print -exec rm -rf {} +
find "$ROOT/vendor" -mindepth 2 -maxdepth 2 -name .git -type f -print -delete
"$PY" "$ROOT/scripts/source_lock.py" verify --strict-hash --config "$ROOT/configs/backbone.yaml"
echo "Vendor .git metadata removed; source-tree verification remains active."
