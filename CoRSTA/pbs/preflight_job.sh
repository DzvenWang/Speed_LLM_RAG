#!/usr/bin/env bash
# Common preflight runner for CPU and optional GPU PBS jobs.
set -euo pipefail
MODE="${1:?Expected cpu or gpu}"
[[ "$MODE" == cpu || "$MODE" == gpu ]] || { echo "Invalid preflight mode: $MODE" >&2; exit 2; }
ROOT="${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-}}"
[[ -d "$ROOT/CoRSTA" ]] || { echo "CoRSTA missing under $ROOT" >&2; exit 2; }
cd "$ROOT"
source CoRSTA/pbs/runtime.sh
: "${TARGET:?TARGET=speed or flow required}"
: "${BACKBONE_CHECKPOINT:?BACKBONE_CHECKPOINT must point to STLA best.pt}"
DATASET="${DATASET:-TraffiDent_Net_06}"
SEED="${SEED:-10}"
export TRAFFIDENT_TARGET="$TARGET"
LOG_DIR="$ROOT/CoRSTA_logs/preflight/$TARGET/${BACKBONE_MODE:-incomplete}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/CoRSTA.preflight.${MODE}.${TARGET}.${DATASET}.seed${SEED}.${PBS_JOBID:-local}.live.log"
ARGS=(--stage preflight --target "$TARGET" --dataset "$DATASET" --seed "$SEED" --backbone-checkpoint "$BACKBONE_CHECKPOINT")
if [[ "$MODE" == cpu ]]; then ARGS+=(--cpu); else ARGS+=(--require-cuda); fi
[[ -z "${RUN_DIR:-}" ]] || ARGS+=(--run-dir "$RUN_DIR")
run_preflight() {
  printf 'CORSTA_PREFLIGHT_PBS_START mode=%s host=%s job=%s\n' "$MODE" "$(hostname)" "${PBS_JOBID:-local}"
  "$PYTHON" -u CoRSTA/scripts/run.py "${ARGS[@]}"
  printf 'CORSTA_PREFLIGHT_PBS_PASS mode=%s job=%s\n' "$MODE" "${PBS_JOBID:-local}"
}
set +e
run_preflight 2>&1 | "$PYTHON" -u CoRSTA/scripts/timestamp_filter.py | tee -a "$LOG"
STATES=("${PIPESTATUS[@]}")
set -e
for code in "${STATES[@]}"; do
  if [[ "$code" != 0 ]]; then
    printf '[%s] CORSTA_PREFLIGHT_PBS_FAILED mode=%s status=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$code" | tee -a "$LOG"
    exit "$code"
  fi
done
