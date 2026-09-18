#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
/scratch/mh20/zw1165/envs/TFP/bin/python scripts/qwen_reasoning_preflight.py --stage development
qsub pbs/qwen_reasoning_dataset.pbs
