#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline; PY=${PYTHON:-python}; cd "$ROOT"
mkdir -p ${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline_logs/out
"$PY" scripts/check_leaf_tfp_runtime.py
"$PY" tests/smoke_leaf_graph_compat.py
"$PY" scripts/validate_cache.py --stage development
"$PY" scripts/verify_official_sources.py
J=$(qsub pbs/methods/leaf_development.pbs); echo "LEAF branches Development seed10 -> $J"
