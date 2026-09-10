#!/bin/bash
set -euo pipefail

ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
cd "$ROOT"

echo "Baseline v0.4.0 submission guide"
echo ""
echo "Runnable before final backbone freeze: GinAR, RAFT, SARAF Development."
echo "GPU smoke is recommended for first-time adapter validation, but it is NOT required for submission."
echo ""
echo "Optional GPU smoke commands:"
echo "  qsub -v METHOD=GinAR pbs/methods/new_main_gpu_smoke.pbs"
echo "  qsub -v METHOD=RAFT  pbs/methods/new_main_gpu_smoke.pbs"
echo "  qsub -v METHOD=SARAF pbs/methods/new_main_gpu_smoke.pbs"
echo ""
echo "Development submission (performs hard cache/source/leakage checks, then only warns about smoke status):"
echo "  bash scripts/submit_prebackbone_main_development.sh"
echo ""
echo "Wait for the final frozen backbone before formal ImputeFormer/RAST matched-backbone runs."
