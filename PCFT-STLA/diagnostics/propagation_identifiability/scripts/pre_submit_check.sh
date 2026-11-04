#!/bin/bash
# Login-node CPU-only pre-submission audit. No GPU requested or required.
set -Eeuo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
cd "$ROOT"
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON_BIN="${TFP_PYTHON:-/g/data/mh20/zw1165/envs/TFP/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "FAIL: TFP Python not executable: $PYTHON_BIN";exit 120; }
[[ -d Backbone && -d Base_SupportBench ]] || { echo 'FAIL: Run from Speed_LLM_RAG root'; exit 121; }
"$PYTHON_BIN" - <<'PY'
import sys,torch,numpy,pandas,yaml
print('CHECK_PYTHON', sys.executable,flush=True)
print('CHECK_TORCH',torch.__version__,flush=True)
PY
bash -n PCFT-STLA/diagnostics/propagation_identifiability/pbs/propagation_diagnostic.pbs
bash -n PCFT-STLA/diagnostics/propagation_identifiability/pbs/stats_only.pbs
# Catch regressions in Gadi's compute-node storage mapping: this was absent
# from v1.2 despite the same directive being present in working Backbone/PCFT v6.
for f in PCFT-STLA/diagnostics/propagation_identifiability/pbs/{propagation_diagnostic,stats_only}.pbs; do
  grep -Fxq '#PBS -l storage=scratch/mh20+gdata/mh20' "$f" || { echo "FAIL: missing PBS project storage request in $f"; exit 124; }
done
echo 'CHECK_PBS_STORAGE scratch/mh20+gdata/mh20 present in both scripts'
"$PYTHON_BIN" -m compileall -q PCFT-STLA/diagnostics/propagation_identifiability
"$PYTHON_BIN" -m pytest -q PCFT-STLA/diagnostics/propagation_identifiability/tests
for target in speed flow; do
  if [[ "$target" == speed ]]; then
    CKPT="Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
  else
    CKPT="Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
  fi
  [[ -f "$CKPT" ]] || { echo "FAIL: missing checkpoint $CKPT";exit 122; }
  "$PYTHON_BIN" -u PCFT-STLA/diagnostics/propagation_identifiability/scripts/run_diagnostic.py \
    --phase preflight --confirm-val --target "$target" --seed 10 \
    --checkpoint "$CKPT"
done
echo 'READY_FOR_GPU_PBS: storage PBS directives, both Speed/Flow CPU preflights, Train/VAL inference and all tests passed'
