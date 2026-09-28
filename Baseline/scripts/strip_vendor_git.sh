#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
PY=${PYTHON:-python}
"$PY" "$ROOT/scripts/source_lock.py" verify --root "$ROOT" --sources "$ROOT/configs/official_sources.yaml"
find "$ROOT/vendor" -mindepth 2 -maxdepth 2 -name .git -type d -prune -print -exec rm -rf {} +
find "$ROOT/vendor" -mindepth 2 -maxdepth 2 -name .git -type f -print -delete
"$PY" "$ROOT/scripts/source_lock.py" verify --root "$ROOT" --sources "$ROOT/configs/official_sources.yaml"
echo "Vendor .git metadata removed; source-tree verification remains active."
