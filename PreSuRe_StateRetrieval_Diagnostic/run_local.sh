#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PRESURE_SRD_PYTHON:-python}"
exec "$PY" scripts/run_local_pipeline.py "$@"
