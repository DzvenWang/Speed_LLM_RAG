#!/bin/bash
set -euo pipefail
cd ${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline
qsub -v SEED=${1:-10} pbs/methods/ginar_stability_probe.pbs
