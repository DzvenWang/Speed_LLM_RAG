#!/usr/bin/env bash
# Submit DIAGNOSTICS + TEST for a v5.3.0 run that has already been finalized.
# Does not submit training or calibration jobs; old failed POST jobs may be replaced.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo 'Usage: bash TCRR-STLA/pbs/submit_postprocess_existing_v5.sh /absolute/path/to/completed/run' >&2
  exit 2
fi
RUN_DIR="$(cd "$1" && pwd -P)"
case "$RUN_DIR" in
  "$ROOT"/TCRR-STLA_exps/*) ;;
  *) echo "ERROR: RUN_DIR is outside TCRR-STLA_exps: $RUN_DIR" >&2; exit 2;;
esac

for f in best_c12.pt best_controller.pt s0_shared_adapter.pt summary.json validation_partitions.json; do
  [[ -s "$RUN_DIR/$f" ]] || { echo "ERROR: Missing formal artifact: $RUN_DIR/$f" >&2; exit 2; }
done

TFP_ENV_ROOT="${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}"
PYTHON="${TCRR_POST_PYTHON:-$TFP_ENV_ROOT/bin/python}"
[[ -x "$PYTHON" ]] || { echo "ERROR: Python not executable: $PYTHON" >&2; exit 2; }

# Validate the identity without loading any dataset or allocating the GPU.
# This detects the original train-batch / evaluation-batch confusion early.
TARGET="$(PYTHONPATH="$ROOT/TCRR-STLA:$ROOT/Backbone:$ROOT/Base_SupportBench:${PYTHONPATH:-}" "$PYTHON" - "$RUN_DIR" <<'PY'
from pathlib import Path
import hashlib, sys, torch
from tcrr_stla.v5.shared_adapter import s0_identity

run=Path(sys.argv[1])
a=torch.load(run/'s0_shared_adapter.pt',map_location='cpu',weights_only=False)
a_sha=hashlib.sha256((run/'s0_shared_adapter.pt').read_bytes()).hexdigest()
assert a.get('version') == '5.3.0', 'S0 version mismatch'
assert not a.get('smoke', False), 'Smoke S0 is not eligible for formal POST'
target=None
for name,stage in [('best_c12.pt','c12'),('best_controller.pt','controller')]:
    c=torch.load(run/name,map_location='cpu',weights_only=False)
    assert c.get('version') == '5.3.0' and c.get('stage') == stage, f'Bad stage/version: {name}'
    assert c.get('adapter_sha256') == a_sha, f'Adapter SHA mismatch: {name}'
    cfg=c['config']
    assert a.get('s0_identity') == s0_identity(cfg), f'S0 formal training identity mismatch: {name}'
    assert cfg['experiment']['target'] == a.get('target'), f'Target mismatch: {name}'
    assert cfg['experiment']['dataset'] == a.get('dataset'), f'Dataset mismatch: {name}'
    if target is None: target=cfg['experiment']['target']
    else: assert target==cfg['experiment']['target'], 'Mixed targets in run'
assert target in ('speed','flow'), 'Invalid target'
print(target)
PY
)"

POST_NCPUS="${TCRR_POST_PBS_NCPUS:-12}"
POST_MEM="${TCRR_POST_PBS_MEM:-32GB}"
POST_WALLTIME="${TCRR_POST_WALLTIME:-48:00:00}"
[[ "$POST_NCPUS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: Bad POST ncpus: $POST_NCPUS" >&2; exit 2; }
[[ "$POST_MEM" =~ ^[1-9][0-9]*(GB|gb)$ ]] || { echo "ERROR: Bad POST mem: $POST_MEM" >&2; exit 2; }
[[ "$POST_WALLTIME" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "ERROR: Bad POST walltime: $POST_WALLTIME" >&2; exit 2; }

POST_ID="$(qsub -N "T53_${TARGET}_pp" \
  -l "ncpus=$POST_NCPUS" -l ngpus=1 -l "mem=$POST_MEM" \
  -l "walltime=$POST_WALLTIME" \
  -v "RUN_DIR=$RUN_DIR,TFP_ENV_ROOT=$TFP_ENV_ROOT" \
  TCRR-STLA/pbs/postprocess_v5.pbs)"
[[ -n "$POST_ID" ]] || { echo 'ERROR: qsub returned no Job ID' >&2; exit 2; }
echo "V53_FORMAL_POST_PREFLIGHT_PASS target=$TARGET training_batch_verified=true"
echo "POSTPROCESS_ID=$POST_ID"
echo "RUN_DIR=$RUN_DIR"
echo "POST_PBS_MEM=$POST_MEM"
echo 'RETRAIN_EPOCHS=0'
