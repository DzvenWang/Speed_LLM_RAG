#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle
STAGE=${STAGE:-development}; SEED=${SEED:-10}; RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_oracle_seed${SEED}}
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle_logs/out /scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle_exps
J=$(qsub -v STAGE="$STAGE",SEED="$SEED",RUN_TAG="$RUN_TAG" "$ROOT/pbs/patchstg_oracle.pbs")
echo "ORACLE_JOB=$J"
echo "RUN_TAG=$RUN_TAG"
echo "RESULT_ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle_exps/TraffiDent_Net_06/${STAGE}/${RUN_TAG}"
echo "FOLLOW=tail -f \$(ls -1t /scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle_logs/PatchSTGOracle.${STAGE}.seed${SEED}.*.gadi-pbs.live.log | head -1)"
