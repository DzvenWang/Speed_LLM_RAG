#!/bin/bash
# Shared live-log bootstrap for Baseline PBS jobs.
# Usage:
#   source "$ROOT/scripts/pbs_live_log.sh"
#   baseline_live_log_start "TaskName" "stage" ["extra.segment" ...]
#
# Produces a continuously updated log under Baseline_logs with:
#   <Task>.<segments>.<YYYYmmdd_HHMMSS>.<PBS_JOBID>.live.log
# Every emitted line is prefixed by timestamp_stream.py.

baseline_live_log_start() {
    if [[ -z "${ROOT:-}" ]]; then
        echo "ERROR: ROOT must be set before baseline_live_log_start" >&2
        return 2
    fi
    if [[ -z "${PY:-}" ]]; then
        echo "ERROR: PY must be set before baseline_live_log_start" >&2
        return 2
    fi

    local task="${1:?task name required}"
    shift || true
    local logroot="${LOGROOT:-${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline_logs}"
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local jobid="${PBS_JOBID:-manual}"
    local name="$task"
    local seg

    for seg in "$@"; do
        [[ -n "$seg" ]] || continue
        # Keep filenames shell-safe and easy to glob.
        seg="${seg//\//_}"
        seg="${seg// /_}"
        name+=".${seg}"
    done

    mkdir -p "$logroot"
    export BASELINE_LOG_STAMP="$stamp"
    export BASELINE_LIVE_LOG="$logroot/${name}.${stamp}.${jobid}.live.log"
    touch "$BASELINE_LIVE_LOG"

    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export BASELINE_TIMESTAMP_STREAM=1

    # From this point onward stdout/stderr is streamed continuously to the live log.
    exec > >("$PY" -u "$ROOT/timestamp_stream.py" | tee -a "$BASELINE_LIVE_LOG") 2>&1

    echo "LIVE_LOG=$BASELINE_LIVE_LOG"
    echo "LOG_STAMP=$BASELINE_LOG_STAMP"
    echo "PBS_JOBID=$jobid"
}

baseline_install_exit_trap() {
    trap 'rc=$?; echo "SCRIPT_EXIT_STATUS=$rc"; exit "$rc"' EXIT
}
