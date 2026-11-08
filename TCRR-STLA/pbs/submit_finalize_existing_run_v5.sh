#!/usr/bin/env bash
# Finalize a v5.3.0 run whose 60 training epochs succeeded, but post-training
# controller calibration failed. Submit one TRAIN-finalization job and one
# dependent combined DIAGNOSTICS+TEST job. Do NOT retrain completed epochs.
# Usage: bash TCRR-STLA/pbs/submit_finalize_existing_run_v5.sh RUN_DIR BACKBONE_CKPT
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 <original RUN_DIR> <original STLAformer best.pt>" >&2; exit 2; }
cd "${SPEED_LLM_RAG_ROOT:-$(pwd)}"
ROOT=$(pwd -P)
RUN_DIR=$(realpath -e -- "$1")
BACKBONE_CHECKPOINT=$(realpath -e -- "$2")
case "$RUN_DIR" in "$ROOT/TCRR-STLA_exps/"*) ;; *) echo "RUN_DIR outside TCRR-STLA_exps: $RUN_DIR" >&2; exit 2;; esac
REL=${RUN_DIR#"$ROOT/TCRR-STLA_exps/"}
IFS=/ read -r TARGET BACKBONE_MODE DATASET VERSION STAGE_DIR RUN_STAMP EXTRA <<< "$REL"
[[ -z "${EXTRA:-}" && "$VERSION" == v5 && "$STAGE_DIR" == formal ]] || { echo "Not a formal v5 run: $REL" >&2; exit 2; }
[[ "$TARGET" == speed || "$TARGET" == flow ]] || { echo "Invalid target: $TARGET" >&2; exit 2; }
[[ "$BACKBONE_MODE" == incomplete || "$BACKBONE_MODE" == complete ]] || { echo "Invalid mode: $BACKBONE_MODE" >&2; exit 2; }
[[ "$DATASET" =~ ^TraffiDent_Net_[0-9]+$ ]] || { echo "Invalid dataset: $DATASET" >&2; exit 2; }
[[ "$RUN_STAMP" =~ ^[A-Za-z0-9_.-]+_seed([0-9]+)_full$ ]] || { echo "Cannot infer seed from run directory: $RUN_STAMP" >&2; exit 2; }
SEED=${BASH_REMATCH[1]}
RUN_STAMP=${RUN_STAMP%_seed${SEED}_full}
case "$BACKBONE_CHECKPOINT" in "$ROOT/Backbone_exps/$TARGET/STLAformer/$DATASET/"*) ;; *) echo 'Incorrect Backbone/Target combination' >&2; exit 2;; esac
case "$BACKBONE_MODE:$BACKBONE_CHECKPOINT" in complete:*/formal_complete/*|incomplete:*/formal/*) ;; *) echo 'Backbone training mode mismatch' >&2; exit 2;; esac
for file in last.pt best_c12.pt best_controller.pt s0_shared_adapter.pt; do
  [[ -s "$RUN_DIR/$file" ]] || { echo "Missing saved file: $RUN_DIR/$file" >&2; exit 2; }
done
# This run has exactly 50 C12 and 10 Controller epochs. Verify the saved
# training state before submitting, to avoid accidental extra training.
TFP_ENV_ROOT=${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}
PYTHON="$TFP_ENV_ROOT/bin/python"
[[ -x "$PYTHON" ]] || { echo "TFP Python not found: $PYTHON" >&2; exit 2; }
"$PYTHON" - "$RUN_DIR/last.pt" <<'PY'
import sys,torch
p=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
if p.get('version')!='5.3.0' or p.get('stage')!='controller' or int(p.get('epoch',-1))!=60:
    raise SystemExit('Not a completed v5.3.0 C3 epoch-60 checkpoint; refusing finalize-only submission')
print('V5_FINALIZE_RESUME_VERIFIED epoch=60 stage=controller version=5.3.0',flush=True)
PY
# Preserve original training batch size: it must match stored run config.
BATCH_SIZE=${BATCH_SIZE:-64}
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] && (( BATCH_SIZE > 0 && BATCH_SIZE <= 512 )) || { echo 'Invalid BATCH_SIZE' >&2;exit 2; }
TRAIN_NCPUS=${TCRR_PBS_NCPUS:-12}
TRAIN_MEM=${TCRR_PBS_MEM:-96GB}
POST_MEM=${TCRR_POST_PBS_MEM:-32GB}
POST_WALLTIME=${TCRR_POST_WALLTIME:-48:00:00}
for mem in "$TRAIN_MEM" "$POST_MEM"; do
  [[ "$mem" =~ ^[1-9][0-9]*(GB|gb)$ ]] || { echo "Invalid memory: $mem" >&2;exit 2; }
done
VARS="TARGET=$TARGET,SEED=$SEED,DATASET=$DATASET,BACKBONE_MODE=$BACKBONE_MODE,BACKBONE_CHECKPOINT=$BACKBONE_CHECKPOINT,TFP_ENV_ROOT=$TFP_ENV_ROOT,RUN_DIR=$RUN_DIR,RUN_STAMP=$RUN_STAMP,BATCH_SIZE=$BATCH_SIZE"
# Crucial: RESUME_CHECKPOINT refers to the old last.pt, so train_v5.py
# will select that SAME run directory and skip 1..60 completed epochs.
FINALIZE_ID=$(qsub -N "T53_${TARGET}_fin" -l "ncpus=$TRAIN_NCPUS" -l "mem=$TRAIN_MEM" -l 'walltime=04:00:00' \
    -v "$VARS,RESUME_CHECKPOINT=$RUN_DIR/last.pt,SKIP_EVALUATE=1,SKIP_DIAGNOSTICS=1" \
    TCRR-STLA/pbs/formal_v5.pbs)
if ! POST_ID=$(qsub -N "T53_${TARGET}_pp" -l "ncpus=$TRAIN_NCPUS" -l "mem=$POST_MEM" -l "walltime=$POST_WALLTIME" \
    -W "depend=afterok:$FINALIZE_ID" -v "$VARS" TCRR-STLA/pbs/postprocess_v5.pbs); then
  echo "POST submission failed. FINALIZE_ID=$FINALIZE_ID already submitted; investigate before retrying." >&2;exit 1
fi
printf 'FINALIZE_ID=%s\nPOSTPROCESS_ID=%s\nRUN_DIR=%s\nRETRAIN_EPOCHS=0\n' "$FINALIZE_ID" "$POST_ID" "$RUN_DIR"
