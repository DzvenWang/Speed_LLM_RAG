#!/bin/bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ROOT="${SPEED_LLM_RAG_ROOT:-$SCRIPT_ROOT}"
ENVROOT="${TFP_ENV_ROOT:-${CONDA_PREFIX:-}}"
[[ -n "$ENVROOT" && -x "$ENVROOT/bin/python" ]] || { echo "ERROR: activate TFP before submitting" >&2; exit 3; }
PBS="$ROOT/Baseline/pbs/methods/formal_primary.pbs"
SEED="${SEED:-10}"
BACKBONE="${BACKBONE:-STLAformer}"
methods=(ImputeFormer LRTC-TMCP GinAR RAFT RAST SARAF ReMFT)
targets=(speed flow)
short_name() {
  case "$1" in
    ImputeFormer) echo Impute;;
    LRTC-TMCP) echo LRTC;;
    GinAR) echo GinAR;;
    RAFT) echo RAFT;;
    RAST) echo RAST;;
    SARAF) echo SARAF;;
    ReMFT) echo ReMFT;;
  esac
}
for method in "${methods[@]}"; do
  for target in "${targets[@]}"; do
    name="$(short_name "$method")_${target}"
    if [[ "$method" == "ImputeFormer" || "$method" == "LRTC-TMCP" ]]; then
      echo "SUBMIT $name METHOD=$method TARGET=$target SEED=$SEED BACKBONE=$BACKBONE"
      qsub -N "$name" -v METHOD="$method",TARGET="$target",SEED="$SEED",BACKBONE="$BACKBONE",TFP_ENV_ROOT="$ENVROOT",SPEED_LLM_RAG_ROOT="$ROOT" "$PBS"
    else
      echo "SUBMIT $name METHOD=$method TARGET=$target SEED=$SEED"
      qsub -N "$name" -v METHOD="$method",TARGET="$target",SEED="$SEED",TFP_ENV_ROOT="$ENVROOT",SPEED_LLM_RAG_ROOT="$ROOT" "$PBS"
    fi
  done
done
