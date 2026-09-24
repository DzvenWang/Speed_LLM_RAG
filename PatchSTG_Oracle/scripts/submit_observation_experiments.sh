#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle
EXP=/scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle_exps/TraffiDent_Net_06
STAGE=${STAGE:-development}; SEED=${SEED:-10}; RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_observation_seed${SEED}}
mkdir -p /scratch/mh20/zw1165/Speed_LLM_RAG/PatchSTG_Oracle_logs/out "$EXP"
J_MASK=$(qsub -v STAGE="$STAGE",SEED="$SEED",RUN_TAG="$RUN_TAG",VARIANT=mask_aware "$ROOT/pbs/patchstg_observation_aware.pbs")
J_AUX=$(qsub -v STAGE="$STAGE",SEED="$SEED",RUN_TAG="$RUN_TAG",VARIANT=state_aux "$ROOT/pbs/patchstg_observation_aware.pbs")
MASK_CKPT="$EXP/$STAGE/$RUN_TAG/mask_aware/best.pt"
AUX_CKPT="$EXP/$STAGE/$RUN_TAG/state_aux/best.pt"
J_CURVE=$(qsub -W depend=afterok:"$J_MASK":"$J_AUX" -v STAGE="$STAGE",SEED="$SEED",RUN_TAG="$RUN_TAG",MASK_AWARE_CKPT="$MASK_CKPT",STATE_AUX_CKPT="$AUX_CKPT" "$ROOT/pbs/patchstg_coverage_curve.pbs")
echo "MASK_AWARE_JOB=$J_MASK"
echo "STATE_AUX_JOB=$J_AUX"
echo "COVERAGE_CURVE_JOB=$J_CURVE"
echo "RUN_TAG=$RUN_TAG"
echo "RESULT_ROOT=$EXP/$STAGE/$RUN_TAG"
