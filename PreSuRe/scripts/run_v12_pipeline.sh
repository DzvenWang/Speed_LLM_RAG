#!/usr/bin/env bash
set -euo pipefail
STAGE="${1:-development}"
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
python scripts/prepare_v12_artifacts.py --stage "$STAGE"
python scripts/train_dual_query_retriever.py --stage "$STAGE"
python scripts/train_v12_forecaster.py --stage "$STAGE"
