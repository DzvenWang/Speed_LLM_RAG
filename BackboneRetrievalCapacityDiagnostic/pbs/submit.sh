#!/bin/bash
set -euo pipefail
TARGET=${1:-${TARGET:-}}
[[ -n "$TARGET" ]] || { echo "Usage: $0 {speed|flow}" >&2; exit 2; }
case "$TARGET" in speed|flow) ;; *) echo "ERROR unsupported TARGET=$TARGET" >&2; exit 2;; esac
SEED=${SEED:-10}
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}
if [[ "$(basename "$ROOT")" == "BackboneRetrievalCapacityDiagnostic" ]]; then ROOT="$(cd "$ROOT/.." && pwd)"; fi
PROJ="$ROOT/BackboneRetrievalCapacityDiagnostic"
ENVROOT=${TFP_ENV_ROOT:-${CONDA_PREFIX:-}}
[[ -n "$ENVROOT" && -x "$ENVROOT/bin/python" ]] || { echo "ERROR activate TFP or export TFP_ENV_ROOT" >&2; exit 3; }
LOGROOT="$ROOT/BackboneRetrievalCapacityDiagnostic_logs/$TARGET"; mkdir -p "$LOGROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
PBSOUT="$LOGROOT/BRCD.${TARGET}.formal.seed${SEED}.${STAMP}.OU"
VARS="TARGET=$TARGET,SEED=$SEED,SPEED_LLM_RAG_ROOT=$ROOT,TFP_ENV_ROOT=$ENVROOT"
JOBID=$(qsub -N "BRCD_${TARGET}" -j oe -o "$PBSOUT" -v "$VARS" "$PROJ/pbs/run_diagnostic.pbs")
echo "SUBMIT_PASS target=$TARGET seed=$SEED job=$JOBID"
echo "PBS_OUTPUT=$PBSOUT"
echo "LIVE_LOG_DIR=$LOGROOT"
