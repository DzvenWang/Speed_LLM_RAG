#!/bin/bash
set -euo pipefail
ROOT=${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
cd "$ROOT"
METHOD=${1:-}
if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 {SPRINT|SLPF|MoGERNN|LEAF|LEAF-Qwen1.5B|RAST-PatchSTG} [seed]"; exit 2
fi
SEED=${2:-10}
case "$METHOD" in
  SPRINT|MoGERNN) qsub -v METHOD="$METHOD",SEED="$SEED" pbs/methods/trc_development.pbs;;
  LEAF) qsub -v SEED="$SEED" pbs/methods/leaf_development.pbs;;
  LEAF-Qwen1.5B) qsub -v SEED="$SEED" pbs/methods/leaf_qwen_development.pbs;;
  SLPF) echo "SLPF remains secondary; use its dedicated runner/PBS after primary experiments.";;
  RAST-PatchSTG) echo "RAST-PatchSTG remains controlled secondary; use the dedicated PatchSTG adaptation after primary experiments.";;
  *) echo "Unknown secondary METHOD=$METHOD"; exit 2;;
esac
