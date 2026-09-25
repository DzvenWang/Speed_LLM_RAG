#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench}"
PYTHON="${PYTHON:-/scratch/mh20/zw1165/envs/TFP/bin/python}"
CFG="${CONFIG:-$ROOT/configs/supportbench.yaml}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

echo "[1/3] Rebuild Net_06 core/layout catalogues for 20/40/60/90"
"$PYTHON" -u build/01_build_core.py --config "$CFG" --subgraph TraffiDent_Net_06
echo "[2/3] Rebuild Net_06 sensing view manifest"
"$PYTHON" -u build/03_build_sensing_views.py --config "$CFG" --subgraph TraffiDent_Net_06
echo "[3/3] Audit Net_06 multi-severity benchmark"
"$PYTHON" -u build/06_audit.py --config "$CFG" --subgraph TraffiDent_Net_06
echo "MULTI-SEVERITY NET_06 BUILD PASS"
