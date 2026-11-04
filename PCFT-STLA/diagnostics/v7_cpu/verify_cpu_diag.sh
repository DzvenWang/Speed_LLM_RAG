#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
REPO="$(cd "$ROOT/.." && pwd -P)"
PYTHON="${PYTHON:-/g/data/mh20/zw1165/envs/TFP/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python)"; fi
export PYTHONPATH="$ROOT:$REPO/Backbone:$REPO/Base_SupportBench:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=''
"$PYTHON" -m pytest -q "$ROOT/tests" "$ROOT/diagnostics/propagation_identifiability/tests" "$ROOT/diagnostics/v7_cpu/tests"
"$PYTHON" -m compileall -q "$ROOT/pcft_stla/v7" "$ROOT/diagnostics/v7_cpu"
"$PYTHON" "$ROOT/diagnostics/v7_cpu/run.py" --help >/dev/null
bash -n "$ROOT/diagnostics/v7_cpu/pbs/cpu_diagnostic.pbs"
echo CPU_DIAGNOSTIC_SOURCE_TESTS_PASS
