#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
PY=${PYTHON:-${PYTHON:-python}}
cd "$ROOT"
if "$PY" scripts/check_leaf_tfp_runtime.py; then exit 0; fi
# Only LEAF's lightweight missing logging dependency is safe to add automatically.
if ! "$PY" - <<'PY' >/dev/null 2>&1
import loguru
PY
then
  "$PY" -m pip install --no-cache-dir --no-deps loguru
fi
"$PY" scripts/check_leaf_tfp_runtime.py
