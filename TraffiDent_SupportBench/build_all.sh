#!/usr/bin/env bash
set -euo pipefail

PROCESS_ROOT="${PROCESS_ROOT:-/scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench}"
CONFIG="${CONFIG:-$PROCESS_ROOT/configs/supportbench.yaml}"
export PYTHONPATH="$PROCESS_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
cd "$PROCESS_ROOT"

stamp() { date '+%F %T'; }
say() {
  # PBS outer timestamping adds exactly one timestamp to every line. Manual
  # execution retains a native timestamp on stage markers.
  if [[ "${SUPPORTBENCH_OUTER_TIMESTAMP:-0}" == "1" ]]; then
    echo "$*"
  else
    echo "[$(stamp)] $*"
  fi
}

run_stage() {
  local number="$1"
  local label="$2"
  shift 2
  local t0 t1 elapsed rc
  t0=$(date +%s)
  say "========== STAGE ${number} START: ${label} =========="
  if "$@"; then
    t1=$(date +%s)
    elapsed=$((t1-t0))
    say "========== STAGE ${number} DONE: ${label} elapsed=${elapsed}s =========="
  else
    rc=$?
    t1=$(date +%s)
    elapsed=$((t1-t0))
    say "========== STAGE ${number} FAILED: ${label} rc=${rc} elapsed=${elapsed}s ==========" >&2
    return "${rc}"
  fi
}

BUILD_T0=$(date +%s)
say "TraffiDent-SupportBench build v$(cat VERSION 2>/dev/null || echo unknown)"
say "PROCESS_ROOT=${PROCESS_ROOT}"
say "CONFIG=${CONFIG}"

run_stage 00 "preflight" python -u preflight.py --config "$CONFIG"
run_stage 01 "canonical core" python -u build/01_build_core.py --config "$CONFIG"
run_stage 02 "forecast windows" python -u build/02_build_windows.py --config "$CONFIG"
run_stage 03 "controlled sensing views" python -u build/03_build_sensing_views.py --config "$CONFIG"
run_stage 04 "train-only support memory" python -u build/04_build_support_memory.py --config "$CONFIG"
run_stage 05 "experimental protocols" python -u build/05_build_protocols.py --config "$CONFIG"
run_stage 06 "benchmark audit" python -u build/06_audit.py --config "$CONFIG"
run_stage 07 "canonical API smoke" python -u build/07_smoke_api.py --config "$CONFIG" --subgraph TraffiDent_Net_06

BUILD_T1=$(date +%s)
say "TraffiDent-SupportBench build completed. total_elapsed=$((BUILD_T1-BUILD_T0))s"
