#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "$ROOT"
PYTHON="${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python)";fi
export PYTHONPATH="$ROOT/PCFT-STLA:$ROOT/Base_SupportBench:$ROOT/Backbone:${PYTHONPATH:-}"
"$PYTHON" -m compileall -q PCFT-STLA/pcft_stla/v7 PCFT-STLA/scripts/run_v7.py
bash -n PCFT-STLA/pbs/formal_v7.pbs PCFT-STLA/pbs/smoke_v7.pbs
for x in formal_v7.pbs smoke_v7.pbs; do
  grep -q '^#PBS -l storage=scratch/mh20+gdata/mh20' "PCFT-STLA/pbs/$x" || { echo "MISSING_GADI_STORAGE $x";exit 2; }
done
"$PYTHON" -m pytest -q PCFT-STLA/tests PCFT-STLA/diagnostics/propagation_identifiability/tests
printf '%s\n' 'V7_SOURCE_TESTS_PASS'
if [[ "${1:-}" == '--preflight' ]];then
 for target in speed flow;do
  if [[ "$target" == speed ]];then CK="Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
  else CK="Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt";fi
  "$PYTHON" -u PCFT-STLA/scripts/run_v7.py --target "$target" --backbone-checkpoint "$CK" --preflight-only
 done
 echo 'V7_READY_FOR_GPU_SMOKE'
fi
