#!/bin/bash
set -euo pipefail
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
qsub -v SEED=${1:-10} pbs/methods/ginar_stability_probe.pbs
