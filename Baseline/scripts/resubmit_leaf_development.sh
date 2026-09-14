#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline; PY=/scratch/mh20/zw1165/envs/TFP/bin/python; cd "$ROOT"
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline_logs/out
"$PY" scripts/check_leaf_tfp_runtime.py
"$PY" tests/smoke_leaf_graph_compat.py
"$PY" scripts/validate_cache.py --stage development
"$PY" scripts/verify_official_sources.py
J=$(qsub pbs/methods/leaf_development.pbs); echo "LEAF branches Development seed10 -> $J"
