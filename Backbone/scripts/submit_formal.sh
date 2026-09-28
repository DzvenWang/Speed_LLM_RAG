#!/bin/bash
# Submit exactly one Formal Backbone job and keep both PBS output and live log
# under Backbone_logs/<target>/<model>/.
set -euo pipefail

MODEL=${1:-${MODEL:-}}
TARGET=${2:-${TARGET:-}}
SEED=${SEED:-10}

[[ -n "$MODEL" ]] || { echo "Usage: $0 MODEL {speed|flow}" >&2; exit 2; }
[[ -n "$TARGET" ]] || { echo "Usage: $0 MODEL {speed|flow}" >&2; exit 2; }
case "$MODEL" in STAEformer|HimNet|BigST|STDN|PatchSTG|STLAformer) ;; *) echo "ERROR unsupported MODEL=$MODEL" >&2; exit 2;; esac
case "$TARGET" in speed|flow) ;; *) echo "ERROR TARGET=$TARGET" >&2; exit 2;; esac
[[ "$SEED" == "10" ]] || { echo "ERROR Formal seed is fixed to 10" >&2; exit 2; }

ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}
if [[ "$(basename "$ROOT")" == "Backbone" ]]; then ROOT="$(cd "$ROOT/.." && pwd)"; fi
PROJ="$ROOT/Backbone"
ENVROOT=${TFP_ENV_ROOT:-${CONDA_PREFIX:-}}
[[ -n "$ENVROOT" && -x "$ENVROOT/bin/python" ]] || { echo "ERROR activate TFP and export TFP_ENV_ROOT=\$CONDA_PREFIX" >&2; exit 3; }
[[ -d "$PROJ" ]] || { echo "ERROR missing $PROJ" >&2; exit 3; }

LOGROOT="$ROOT/Backbone_logs/$TARGET/$MODEL"
mkdir -p "$LOGROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
PBSOUT="$LOGROOT/${MODEL}.${TARGET}.formal.seed${SEED}.${STAMP}.OU"

COMMON="MODEL=$MODEL,TARGET=$TARGET,SEED=$SEED,SPEED_LLM_RAG_ROOT=$ROOT,TFP_ENV_ROOT=$ENVROOT"
if [[ "$MODEL" == "PatchSTG" ]]; then
  JOBID=$(qsub -N "Patch_${TARGET}" -j oe -o "$PBSOUT" \
    -v "TARGET=$TARGET,STAGE=formal,SEED=$SEED,SPEED_LLM_RAG_ROOT=$ROOT,TFP_ENV_ROOT=$ENVROOT" \
    "$PROJ/pbs/run_patchstg.pbs")
else
  JOBID=$(qsub -N "${MODEL}_${TARGET}" -j oe -o "$PBSOUT" \
    -v "$COMMON" "$PROJ/pbs/run_model_formal.pbs")
fi

echo "SUBMIT_PASS model=$MODEL target=$TARGET seed=$SEED job=$JOBID"
echo "PBS_OUTPUT=$PBSOUT"
echo "LIVE_LOG_DIR=$LOGROOT"
