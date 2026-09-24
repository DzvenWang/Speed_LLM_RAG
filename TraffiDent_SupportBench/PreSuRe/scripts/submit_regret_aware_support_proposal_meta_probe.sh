#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
python scripts/proposal_meta_ranker_preflight.py --stage development
qsub pbs/regret_aware_support_proposal_meta_probe.pbs
