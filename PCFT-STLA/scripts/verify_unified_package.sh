#!/usr/bin/env bash
# Run from Speed_LLM_RAG root; no GPU is required.
set -Eeuo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
cd "$ROOT"
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/PCFT-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON_BIN="${TFP_PYTHON:-/g/data/mh20/zw1165/envs/TFP/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: Python unavailable at $PYTHON_BIN" >&2; exit 2; }
for d in PCFT-STLA/pcft_stla/v6 PCFT-STLA/diagnostics/propagation_identifiability Backbone Base_SupportBench; do
  [[ -d "$d" ]] || { echo "FAIL: missing directory $d" >&2; exit 3; }
done
for f in PCFT-STLA/pbs/formal_v6.pbs PCFT-STLA/diagnostics/propagation_identifiability/pbs/propagation_diagnostic.pbs PCFT-STLA/diagnostics/propagation_identifiability/pbs/stats_only.pbs; do
  bash -n "$f"
  grep -Fxq '#PBS -l storage=scratch/mh20+gdata/mh20' "$f" || { echo "FAIL: PBS storage directive missing in $f" >&2; exit 4; }
done
"$PYTHON_BIN" -m compileall -q PCFT-STLA/pcft_stla/v6 PCFT-STLA/scripts PCFT-STLA/diagnostics/propagation_identifiability
"$PYTHON_BIN" -m pytest -q PCFT-STLA/tests PCFT-STLA/diagnostics/propagation_identifiability/tests
echo 'UNIFIED_SOURCE_TESTS_PASS: v6 + propagation diagnostic'
if [[ "${1:-}" == '--preflight' ]]; then
  for target in speed flow; do
    if [[ "$target" == speed ]]; then
      ckpt='Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt'
    else
      ckpt='Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt'
    fi
    [[ -f "$ckpt" ]] || { echo "FAIL: missing checkpoint $ckpt" >&2; exit 5; }
    "$PYTHON_BIN" -u PCFT-STLA/scripts/preflight_v6.py --target "$target" --dataset TraffiDent_Net_06 --seed 10 --backbone-checkpoint "$ckpt"
  done
  bash PCFT-STLA/diagnostics/propagation_identifiability/scripts/pre_submit_check.sh
  echo 'READY_FOR_GPU_PBS: both v6 and diagnostic real-data CPU preflights passed'
fi
