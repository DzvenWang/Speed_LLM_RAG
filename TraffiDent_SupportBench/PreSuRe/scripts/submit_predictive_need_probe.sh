#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
cd "$ROOT"
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/out /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_exps
"$PY" scripts/probe_preflight.py --stage development
J=$(qsub -v STAGE=development,SEED=10 pbs/predictive_need_probe.pbs)
echo "PreSuRe Predictive Need Probe -> $J"
echo "Reuses completed v0.7 artifacts; does NOT rerun Oracle A-F or M0-M5."
echo "Decision output: PreSuRe_exps/PredictiveNeedProbe/TraffiDent_Net_06/development/LATEST.json"
