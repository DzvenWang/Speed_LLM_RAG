#!/usr/bin/env bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
LOGROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs
OUTROOT="$LOGROOT/out"
STAGE=${STAGE:-development}; SEED=${SEED:-10}
cd "$ROOT"; mkdir -p "$LOGROOT" "$OUTROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
JOBID=$(qsub -v "LOG_STAMP=$STAMP,STAGE=$STAGE,SEED=$SEED" pbs/presure_v12_development.pbs)
JOBNUM="${JOBID%%.*}"
LIVE_LOG="$LOGROOT/PreSuReV12.${STAGE}.seed${SEED}.${STAMP}.${JOBNUM}.gadi-pbs.live.log"
printf 'JOBID=%s\n' "$JOBID"
printf 'LIVE_LOG=%s\n' "$LIVE_LOG"
printf 'PBS_OU_DIR=%s\n' "$OUTROOT"
printf 'FOLLOW=tail -f %s\n' "$LIVE_LOG"
