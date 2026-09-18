#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
cd "$ROOT"
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/out /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_exps

"$PY" scripts/preflight.py --stage development

J=$(qsub -v STAGE=development,SEED=10 pbs/full_pipeline.pbs)
echo "PreSuRe Development FullPipeline (one V100, all phases) -> $J"
echo "Order: preflight -> backbone artifacts -> retrieval/node features -> base Oracles + E24 -> M0/M1/M2/M3 -> M3-Top4 + SoftMix diagnostics -> M4/M5 soft distillation models -> summary"
echo "Only one PBS GPU job is submitted."
