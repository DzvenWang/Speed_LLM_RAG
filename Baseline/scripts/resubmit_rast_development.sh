#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline; PY=/scratch/mh20/zw1165/envs/TFP/bin/python; cd "$ROOT"
"$PY" scripts/check_rast_runtime.py
"$PY" tests/smoke_rast_import_contract.py
"$PY" scripts/validate_cache.py --stage development
"$PY" scripts/verify_official_sources.py
"$PY" scripts/retrieval_leakage_audit.py --stage development
J=$(qsub pbs/methods/rast_development.pbs); echo "RAST-native Development seed10 -> $J"
