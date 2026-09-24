#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
python scripts/deployment_aligned_router_preflight.py
qsub pbs/deployment_aligned_utility_router_probe.pbs
