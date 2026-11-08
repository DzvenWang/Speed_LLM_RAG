#!/usr/bin/env bash
# Submit a read-only mechanism audit. No training / checkpoint rewrites.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo 'Usage: bash TCRR-STLA/pbs/submit_p0_deep_v5.sh /absolute/path/to/v5.3/run' >&2
  exit 2
fi
RUN_DIR="$(cd "$1" && pwd -P)"
case "$RUN_DIR" in "$ROOT"/TCRR-STLA_exps/*) ;; *) echo 'RUN_DIR outside experiment root' >&2; exit 2;; esac
for f in best_controller.pt best_c12.pt s0_shared_adapter.pt summary.json validation_partitions.json; do
  [[ -s "$RUN_DIR/$f" ]] || { echo "P0_PREFLIGHT_FAILED missing=$f" >&2; exit 2; }
done

TFP_ENV_ROOT="${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}"
P0_PYTHON="${P0_PYTHON:-$TFP_ENV_ROOT/bin/python}"
[[ -x "$P0_PYTHON" ]] || { echo "Python not executable: $P0_PYTHON" >&2; exit 2; }
TARGET="$(PYTHONPATH="$ROOT/TCRR-STLA:$ROOT/Backbone:$ROOT/Base_SupportBench:${PYTHONPATH:-}" "$P0_PYTHON" - "$RUN_DIR" <<'PY'
import hashlib,sys,torch
from pathlib import Path
from tcrr_stla.v5.shared_adapter import s0_identity
run=Path(sys.argv[1]);p=run/'best_controller.pt'
ck=torch.load(p,map_location='cpu',weights_only=False)
a=torch.load(run/'s0_shared_adapter.pt',map_location='cpu',weights_only=False)
assert ck.get('version')=='5.3.0' and ck.get('stage')=='controller','Controller v5.3 checkpoint required'
assert ck.get('adapter_sha256')==hashlib.sha256((run/'s0_shared_adapter.pt').read_bytes()).hexdigest(),'S0 SHA mismatch'
assert a.get('s0_identity')==s0_identity(ck['config']),'S0 identity mismatch'
assert ck['config']['experiment']['target'] in ('speed','flow'),'Bad target'
assert a['target']==ck['config']['experiment']['target'],'Target mismatch'
assert (run/'summary.json').stat().st_size>0,'Missing finalized summary'
print(ck['config']['experiment']['target'])
PY
)"

P0_BATCH_SIZE="${P0_BATCH_SIZE:-8}"
P0_MAX_BATCHES="${P0_MAX_BATCHES:-0}"
P0_VALIDATION_SECTION="${P0_VALIDATION_SECTION:-selection}"
P0_RUN_TEST="${P0_RUN_TEST:-1}"
for v in "$P0_BATCH_SIZE" "$P0_MAX_BATCHES"; do [[ "$v" =~ ^[0-9]+$ ]] || { echo 'Bad batch count' >&2; exit 2; }; done
[[ "$P0_BATCH_SIZE" -gt 0 ]] || { echo 'Batch must be positive' >&2; exit 2; }
[[ "$P0_VALIDATION_SECTION" == selection || "$P0_VALIDATION_SECTION" == calibration || "$P0_VALIDATION_SECTION" == check ]] || { echo 'Bad section' >&2; exit 2; }
[[ "$P0_RUN_TEST" == 0 || "$P0_RUN_TEST" == 1 ]] || { echo 'P0_RUN_TEST must be 0/1' >&2; exit 2; }

JOB="$(qsub -N "T53_${TARGET}_P0" \
  -v "RUN_DIR=$RUN_DIR,TFP_ENV_ROOT=$TFP_ENV_ROOT,P0_BATCH_SIZE=$P0_BATCH_SIZE,P0_MAX_BATCHES=$P0_MAX_BATCHES,P0_VALIDATION_SECTION=$P0_VALIDATION_SECTION,P0_RUN_TEST=$P0_RUN_TEST" \
  TCRR-STLA/pbs/p0_deep_diagnostics_v5.pbs)"
[[ -n "$JOB" ]] || { echo 'qsub returned no job ID' >&2; exit 2; }
echo "P0_PREFLIGHT_PASS target=$TARGET read_only=true"
echo "P0_JOB_ID=$JOB"
echo "RUN_DIR=$RUN_DIR"
echo "P0_RUN_TEST=$P0_RUN_TEST"
echo 'RETRAIN_EPOCHS=0'
