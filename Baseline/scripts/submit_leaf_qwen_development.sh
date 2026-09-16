#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline; PY=/scratch/mh20/zw1165/envs/TFP/bin/python
cd "$ROOT"
"$PY" scripts/check_leaf_tfp_runtime.py
"$PY" scripts/check_leaf_qwen_runtime.py
"$PY" tests/smoke_leaf_qwen_contract.py
"$PY" scripts/validate_cache.py --stage development
"$PY" scripts/verify_official_sources.py
PTR=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline_exps/LEAF/TraffiDent_Net_06/development/LATEST_BRANCH_RUN.txt
[[ -s "$PTR" ]] || { echo "ERROR: completed LEAF Development branches are required: $PTR"; exit 1; }
if [[ -n "${LEAF_RESUME_DIR:-}" ]]; then
  jid=$(qsub -v LEAF_RESUME_DIR="$LEAF_RESUME_DIR" pbs/methods/leaf_qwen_development.pbs)
else
  jid=$(qsub pbs/methods/leaf_qwen_development.pbs)
fi
echo "LEAF-Qwen1.5B Development seed10 -> $jid"
