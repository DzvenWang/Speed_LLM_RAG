#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
PY=${PYTHON:-/scratch/mh20/zw1165/envs/TFP/bin/python}
cd "$ROOT"
if "$PY" scripts/check_rast_runtime.py >/dev/null 2>&1; then "$PY" scripts/check_rast_runtime.py; exit 0; fi
"$PY" -m pip install --no-cache-dir --no-deps "${RAST_FAISS_SPEC:-faiss-gpu==1.15.0}"
"$PY" scripts/check_rast_runtime.py
