#!/bin/bash
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
PBS="$ROOT/Baseline/pbs/methods/formal_primary.pbs"
SEED="${SEED:-10}"
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
    echo "SUBMIT $name METHOD=$method TARGET=$target SEED=$SEED"
    qsub -N "$name" -v METHOD="$method",TARGET="$target",SEED="$SEED" "$PBS"
  done
done
