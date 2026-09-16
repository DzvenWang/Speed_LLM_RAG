#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline; PY=/scratch/mh20/zw1165/envs/TFP/bin/python
SEED=${1:-10}
case "$SEED" in 10|20|30) ;; *) echo "ERROR: formal seed must be 10, 20, or 30"; exit 2;; esac
cd "$ROOT"
"$PY" scripts/check_leaf_tfp_runtime.py
"$PY" scripts/check_leaf_qwen_runtime.py
"$PY" tests/smoke_leaf_qwen_contract.py
"$PY" scripts/validate_cache.py --stage formal
"$PY" scripts/verify_official_sources.py
BASE=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline_exps/LEAF-branches/TraffiDent_Net_06/formal
ls -d "$BASE"/*_seed"$SEED" >/dev/null 2>&1 || { echo "ERROR: formal LEAF branch run for seed=$SEED not found under $BASE"; exit 1; }
if [[ -n "${LEAF_RESUME_DIR:-}" ]]; then
  jid=$(qsub -v SEED="$SEED",LEAF_RESUME_DIR="$LEAF_RESUME_DIR" pbs/methods/leaf_qwen_formal.pbs)
else
  jid=$(qsub -v SEED="$SEED" pbs/methods/leaf_qwen_formal.pbs)
fi
echo "LEAF-Qwen1.5B Formal seed${SEED} -> $jid"
