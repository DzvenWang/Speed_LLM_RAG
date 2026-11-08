#!/usr/bin/env bash
# Preserve an already queued/running TRAIN job; submit one combined POST job.
# Usage: bash TCRR-STLA/pbs/submit_post_for_existing_train_v5.sh 182459670.gadi-pbs
# Does NOT cancel legacy held DIAGNOSTICS / TEST jobs; cancel those separately.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: $0 <existing TRAIN PBS Job ID>" >&2; exit 2; }
TRAIN_ID=$1
[[ "$TRAIN_ID" =~ ^[0-9]+(\.gadi-pbs)?$ ]] || { echo "Invalid TRAIN_ID=$TRAIN_ID" >&2; exit 2; }
cd "${SPEED_LLM_RAG_ROOT:-$(pwd)}"
ROOT=$(pwd -P)
[[ -f "$ROOT/TCRR-STLA/pbs/postprocess_v5.pbs" ]] || { echo 'Missing postprocess_v5.pbs' >&2; exit 2; }
TFP_ENV_ROOT=${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}
TCRR_POST_PBS_MEM=${TCRR_POST_PBS_MEM:-32GB}
TCRR_POST_WALLTIME=${TCRR_POST_WALLTIME:-48:00:00}
[[ "$TCRR_POST_PBS_MEM" =~ ^([1-9][0-9]*)(GB|gb)$ ]] || { echo 'Invalid TCRR_POST_PBS_MEM' >&2; exit 2; }
(( 10#${BASH_REMATCH[1]} <= 382 )) || { echo 'TCRR_POST_PBS_MEM exceeds limit' >&2; exit 2; }
[[ "$TCRR_POST_WALLTIME" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]] || { echo 'Invalid TCRR_POST_WALLTIME' >&2; exit 2; }
(( 10#${BASH_REMATCH[1]} >= 1 && 10#${BASH_REMATCH[1]} <= 48 )) || { echo 'TCRR_POST_WALLTIME out of range' >&2; exit 2; }

# Prefer authoritative PBS JSON metadata. A caller can supply RUN_DIR explicitly
# if their PBS installation does not support qstat JSON formatting.
if [[ -z "${RUN_DIR:-}" ]]; then
  if ! METADATA=$(qstat -f -F json "$TRAIN_ID" 2>/dev/null); then
    echo 'Unable to retrieve PBS JSON; set RUN_DIR to the exact value used by TRAIN and retry.' >&2
    exit 2
  fi
  RUN_DIR=$(printf '%s' "$METADATA" | python -c '
import json,sys
try:
    jobs=json.load(sys.stdin)["Jobs"]
    item=next(iter(jobs.values()))
    env=item.get("Variable_List",{})
    if isinstance(env,str):
        env=dict(piece.split("=",1) for piece in env.split(",") if "=" in piece)
    print(env.get("RUN_DIR",""))
except Exception:
    sys.exit(2)
') || { echo 'Cannot decode PBS RUN_DIR; set RUN_DIR explicitly.' >&2; exit 2; }
fi
[[ -n "$RUN_DIR" && "$RUN_DIR" == "$ROOT/TCRR-STLA_exps/"* ]] || {
  echo "RUN_DIR must refer to an existing TCRR run under $ROOT/TCRR-STLA_exps; got: $RUN_DIR" >&2; exit 2;
}
[[ "$RUN_DIR" != *','* && "$RUN_DIR" != *$'\n'* ]] || { echo 'Invalid RUN_DIR' >&2; exit 2; }
case "$RUN_DIR" in
  */speed/*) TARGET=speed ;;
  */flow/*) TARGET=flow ;;
  *) echo 'Cannot infer speed/flow from RUN_DIR' >&2; exit 2 ;;
esac
[[ "$TFP_ENV_ROOT" != *','* ]] || { echo 'Invalid TFP_ENV_ROOT' >&2; exit 2; }
POST_ID=$(qsub -N "T53_${TARGET}_pp" -l ncpus=12 -l "mem=$TCRR_POST_PBS_MEM" -l "walltime=$TCRR_POST_WALLTIME" \
  -W "depend=afterok:$TRAIN_ID" -v "RUN_DIR=$RUN_DIR,TFP_ENV_ROOT=$TFP_ENV_ROOT" \
  TCRR-STLA/pbs/postprocess_v5.pbs)
printf 'PRESERVED_TRAIN_ID=%s\nPOSTPROCESS_ID=%s\nRUN_DIR=%s\nPOST_PBS_MEM=%s\n' \
  "$TRAIN_ID" "$POST_ID" "$RUN_DIR" "$TCRR_POST_PBS_MEM"
echo 'IMPORTANT: cancel the legacy held diagnostics and test PBS jobs only after verifying this new POSTPROCESS_ID.'
