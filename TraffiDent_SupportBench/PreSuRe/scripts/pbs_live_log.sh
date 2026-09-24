#!/bin/bash
presure_live_log_start() {
  local task="${1:?task}"; local stage="${2:?stage}"; local extra="${3:-}"
  local stamp="$(date +%Y%m%d_%H%M%S)"; local jobid="${PBS_JOBID:-manual}"; local name="$task.$stage"
  [[ -n "$extra" ]] && name+=".$extra"
  mkdir -p "$LOGROOT" "$LOGROOT/out"
  export PRESURE_LIVE_LOG="$LOGROOT/${name}.${stamp}.${jobid}.live.log"
  touch "$PRESURE_LIVE_LOG"; export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  exec > >("$PY" -u "$ROOT/timestamp_stream.py" | tee -a "$PRESURE_LIVE_LOG") 2>&1
  echo "LIVE_LOG=$PRESURE_LIVE_LOG"; echo "PBS_JOBID=$jobid"; echo "LOG_STAMP=$stamp"
}
presure_install_exit_trap(){ trap 'rc=$?; echo "SCRIPT_EXIT_STATUS=$rc"; exit "$rc"' EXIT; }
