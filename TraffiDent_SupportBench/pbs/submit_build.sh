#!/usr/bin/env bash
set -euo pipefail

PROCESS_ROOT="/scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench"
cd "$PROCESS_ROOT"

JOB_ID="$(qsub pbs/build_supportbench.pbs)"
LIVE_LOG="$PROCESS_ROOT/logs/SupportBenchBuild.${JOB_ID}.live.log"
echo "Submitted : $JOB_ID"
echo "qstat     : qstat -u $USER"
echo "details   : qstat -fx $JOB_ID"
echo "live log  : $LIVE_LOG"
echo "follow    : tail -f '$LIVE_LOG'"
