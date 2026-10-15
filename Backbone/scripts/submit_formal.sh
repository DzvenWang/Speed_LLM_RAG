#!/bin/bash
# Submit exactly one Formal Backbone job and keep both PBS output and live log
# under Backbone_logs/<target>/<model>/.
set -euo pipefail

MODEL=${1:-${MODEL:-}}
TARGET=${2:-${TARGET:-}}
SEED=${SEED:-10}
TRAIN_DATA_MODE=${TRAIN_DATA_MODE:-incomplete}

[[ -n "$MODEL" ]] || { echo "Usage: $0 MODEL {speed|flow}" >&2; exit 2; }
[[ -n "$TARGET" ]] || { echo "Usage: $0 MODEL {speed|flow}" >&2; exit 2; }
case "$MODEL" in STAEformer|HimNet|BigST|STDN|PatchSTG|STLAformer) ;; *) echo "ERROR unsupported MODEL=$MODEL" >&2; exit 2;; esac
case "$TARGET" in speed|flow) ;; *) echo "ERROR TARGET=$TARGET" >&2; exit 2;; esac
[[ "$SEED" == "10" ]] || { echo "ERROR Formal seed is fixed to 10" >&2; exit 2; }
case "$TRAIN_DATA_MODE" in incomplete|complete) ;; *) echo "ERROR invalid TRAIN_DATA_MODE=$TRAIN_DATA_MODE" >&2; exit 2;; esac

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ROOT=${SPEED_LLM_RAG_ROOT:-$SCRIPT_ROOT}
if [[ "$(basename "$ROOT")" == "Backbone" ]]; then ROOT="$(cd "$ROOT/.." && pwd)"; fi
PROJ="$ROOT/Backbone"
ENVROOT=${TFP_ENV_ROOT:-${CONDA_PREFIX:-}}
[[ -n "$ENVROOT" && -x "$ENVROOT/bin/python" ]] || { echo "ERROR activate TFP and export TFP_ENV_ROOT=\$CONDA_PREFIX" >&2; exit 3; }
[[ -d "$PROJ" ]] || { echo "ERROR missing $PROJ" >&2; exit 3; }

LOGROOT="$ROOT/Backbone_logs/$TARGET/$MODEL"
mkdir -p "$LOGROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
PBSOUT="$LOGROOT/${MODEL}.${TARGET}.formal_${TRAIN_DATA_MODE}.seed${SEED}.${STAMP}.OU"

COMMON="MODEL=$MODEL,TARGET=$TARGET,SEED=$SEED,TRAIN_DATA_MODE=$TRAIN_DATA_MODE,SPEED_LLM_RAG_ROOT=$ROOT,TFP_ENV_ROOT=$ENVROOT"
JOBID=$(qsub -N "${MODEL}_${TARGET}" -j oe -o "$PBSOUT" \
  -v "$COMMON" "$PROJ/pbs/run_model_formal.pbs")

echo "SUBMIT_PASS model=$MODEL target=$TARGET seed=$SEED train_data_mode=$TRAIN_DATA_MODE job=$JOBID"
echo "PBS_OUTPUT=$PBSOUT"
echo "LIVE_LOG_DIR=$LOGROOT"
