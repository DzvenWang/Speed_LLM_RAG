#!/bin/bash
set -euo pipefail

ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
LOGROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline_logs
PY=/scratch/mh20/zw1165/envs/TFP/bin/python

cd "$ROOT"

# Hard gates that define experiment validity.
"$PY" scripts/validate_cache.py --stage development
"$PY" scripts/verify_official_sources.py
"$PY" scripts/retrieval_leakage_audit.py --stage development

# GPU smoke is advisory only.  We report whether a successful smoke exists,
# but never block Development submission because a smoke was skipped/failed.
echo "GPU smoke check (advisory only; not a submission gate):"
for m in GinAR RAFT SARAF; do
  log=$(ls -t "$LOGROOT"/${m}.gpu_smoke.*.live.log 2>/dev/null | head -1 || true)

  if [[ -z "$log" ]]; then
    echo "WARNING GPU_SMOKE_NOT_FOUND method=$m -- proceeding with Development submission."
    continue
  fi

  if grep -q "NEW MAIN GPU SMOKE PASS method=$m" "$log" \
     && grep -q "SCRIPT_EXIT_STATUS=0" "$log"; then
    echo "PASS GPU_SMOKE method=$m log=$log"
  else
    echo "WARNING GPU_SMOKE_NOT_SUCCESSFUL method=$m log=$log -- proceeding with Development submission."
  fi
done

echo "GPU smoke check complete. Smoke is recommended but not required."

echo "Submitting pre-backbone main-table Development runs:"
echo "GinAR -> $(qsub pbs/methods/ginar_development.pbs)"
echo "RAFT  -> $(qsub pbs/methods/raft_development.pbs)"
echo "SARAF -> $(qsub pbs/methods/saraf_development.pbs)"
