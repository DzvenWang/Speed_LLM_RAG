#!/bin/bash
set -euo pipefail
LOGROOT=${LOGROOT:-${SPEED_LLM_RAG_ROOT:-${PBS_O_WORKDIR:-$PWD}}/Baseline_logs}
PATTERN=${1:-*.live.log}
latest=$(find "$LOGROOT" -maxdepth 1 -type f -name "$PATTERN" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
if [[ -z "$latest" ]]; then
  echo "No live log found: $LOGROOT/$PATTERN" >&2
  exit 1
fi
echo "$latest"
if [[ "${TAIL:-0}" == "1" ]]; then
  exec tail -f "$latest"
fi
