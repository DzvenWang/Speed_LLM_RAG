#!/usr/bin/env bash
# Submit formal TRAIN -> combined validation DIAGNOSTICS + locked TEST.
# SMOKE=1 submits only one isolated smoke job; optional full-batch stress mode.
set -euo pipefail
: "${TARGET:?Set TARGET=speed or flow}"
: "${BACKBONE_CHECKPOINT:?Set absolute path to STLAformer best.pt}"
case "$TARGET" in speed|flow) ;; *) echo "Unknown TARGET=$TARGET" >&2; exit 2;; esac
case "${BACKBONE_MODE:-incomplete}" in incomplete|complete) ;; *) echo "Invalid BACKBONE_MODE" >&2; exit 2;; esac
cd "${SPEED_LLM_RAG_ROOT:-$(pwd)}"
ROOT=$(pwd -P)
: "${TFP_ENV_ROOT:=/g/data/mh20/zw1165/envs/TFP}"
SEED=${SEED:-10}
DATASET=${DATASET:-TraffiDent_Net_06}
BACKBONE_MODE=${BACKBONE_MODE:-incomplete}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
SMOKE=${SMOKE:-0}
SMOKE_PROFILE_FULL_BATCH=${SMOKE_PROFILE_FULL_BATCH:-0}
# Speed-first default profile. Every setting is overridable by the caller:
# BATCH_SIZE=32 TCRR_TRAIN_WORKERS=0 TCRR_PBS_MEM=32GB bash ...
# Parameter validation and qsub propagation below remain authoritative.
BATCH_SIZE=${BATCH_SIZE:-64}
TCRR_TRAIN_WORKERS=${TCRR_TRAIN_WORKERS:-4}
TCRR_VAL_WORKERS=${TCRR_VAL_WORKERS:-0}
TCRR_PERSISTENT_WORKERS=${TCRR_PERSISTENT_WORKERS:-1}
TCRR_PREFETCH_FACTOR=${TCRR_PREFETCH_FACTOR:-2}
# Optional qsub resource overrides; they are NOT forwarded as model settings.
# gpuvolta commonly allocates 12 CPU cores per V100. Avoid requesting extra
# CPUs unless the queue accepts and the measured dataloader scaling justifies it.
TCRR_PBS_NCPUS=${TCRR_PBS_NCPUS:-12}
TCRR_PBS_MEM=${TCRR_PBS_MEM:-96GB}
# Independent allocation for combined postprocessing: diagnostics then test.
TCRR_POST_PBS_MEM=${TCRR_POST_PBS_MEM:-32GB}
TCRR_POST_WALLTIME=${TCRR_POST_WALLTIME:-48:00:00}

valid_int() {
  local name="$1" val="$2" min="$3" max="$4"
  [[ "$val" =~ ^[0-9]+$ ]] && (( 10#$val >= min && 10#$val <= max )) || {
    echo "Invalid $name=$val (expected integer in [$min,$max])" >&2; return 2;
  }
}
valid_int SEED "$SEED" 0 2147483647
valid_int SMOKE "$SMOKE" 0 1
valid_int SMOKE_PROFILE_FULL_BATCH "$SMOKE_PROFILE_FULL_BATCH" 0 1
valid_int TCRR_PBS_NCPUS "$TCRR_PBS_NCPUS" 12 48
(( TCRR_PBS_NCPUS % 12 == 0 )) || { echo 'TCRR_PBS_NCPUS must be a multiple of 12 on gpuvolta' >&2; exit 2; }
[[ "$TCRR_PBS_MEM" =~ ^([1-9][0-9]*)(GB|gb)$ ]] || { echo 'Invalid TCRR_PBS_MEM (use e.g. 96GB)' >&2; exit 2; }
(( 10#${BASH_REMATCH[1]} <= 382 )) || { echo 'TCRR_PBS_MEM exceeds single-node gpuvolta maximum' >&2; exit 2; }
[[ "$TCRR_POST_PBS_MEM" =~ ^([1-9][0-9]*)(GB|gb)$ ]] || { echo 'Invalid TCRR_POST_PBS_MEM (use e.g. 32GB)' >&2; exit 2; }
(( 10#${BASH_REMATCH[1]} <= 382 )) || { echo 'TCRR_POST_PBS_MEM exceeds single-node gpuvolta maximum' >&2; exit 2; }
[[ "$TCRR_POST_WALLTIME" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]] || { echo 'Invalid TCRR_POST_WALLTIME (HH:MM:SS)' >&2; exit 2; }
(( 10#${BASH_REMATCH[1]} > 0 && 10#${BASH_REMATCH[1]} <= 48 )) || { echo 'TCRR_POST_WALLTIME must be 01:00:00 to 48:59:59 (subject to queue policy)' >&2; exit 2; }
if (( TCRR_PBS_NCPUS > 12 )); then
  echo "WARNING: requesting ${TCRR_PBS_NCPUS} CPU cores with one V100; confirm Gadi gpuvolta accepts this layout and it actually speeds training" >&2
fi
[[ "$SMOKE_PROFILE_FULL_BATCH" == 0 || "$SMOKE" == 1 ]] || {
  echo 'SMOKE_PROFILE_FULL_BATCH=1 requires SMOKE=1' >&2; exit 2;
}
# Validate all values before calling qsub; no silently dropped or malformed flags.
[[ -z "${BATCH_SIZE:-}" ]] || valid_int BATCH_SIZE "$BATCH_SIZE" 1 512
[[ -z "${TCRR_TRAIN_WORKERS:-}" ]] || valid_int TCRR_TRAIN_WORKERS "$TCRR_TRAIN_WORKERS" 0 8
[[ -z "${TCRR_VAL_WORKERS:-}" ]] || valid_int TCRR_VAL_WORKERS "$TCRR_VAL_WORKERS" 0 8
[[ -z "${TCRR_EVAL_WORKERS:-}" ]] || valid_int TCRR_EVAL_WORKERS "$TCRR_EVAL_WORKERS" 0 8
[[ -z "${TCRR_PERSISTENT_WORKERS:-}" ]] || valid_int TCRR_PERSISTENT_WORKERS "$TCRR_PERSISTENT_WORKERS" 0 1
[[ -z "${TCRR_PREFETCH_FACTOR:-}" ]] || valid_int TCRR_PREFETCH_FACTOR "$TCRR_PREFETCH_FACTOR" 1 8
[[ -z "${DIAGNOSTIC_BATCH_SIZE:-}" ]] || valid_int DIAGNOSTIC_BATCH_SIZE "$DIAGNOSTIC_BATCH_SIZE" 1 512
[[ -z "${DIAGNOSTIC_MAX_BATCHES:-}" ]] || valid_int DIAGNOSTIC_MAX_BATCHES "$DIAGNOSTIC_MAX_BATCHES" 0 1000000
[[ -z "${DIAGNOSTIC_PRIMARY_MAX_BATCHES:-}" ]] || valid_int DIAGNOSTIC_PRIMARY_MAX_BATCHES "$DIAGNOSTIC_PRIMARY_MAX_BATCHES" 0 1000000
for text_field in "$TARGET" "$DATASET" "$BACKBONE_MODE" "$BACKBONE_CHECKPOINT" "$TFP_ENV_ROOT" "$RUN_STAMP" "$ROOT"; do
  [[ "$text_field" != *','* && "$text_field" != *$'\n'* && "$text_field" != *$'\r'* ]] || {
    echo 'PBS -v variable contains a comma or line break' >&2; exit 2;
  }
done
case "$RUN_STAMP" in *[!a-zA-Z0-9_.-]*|'') echo "Invalid RUN_STAMP=$RUN_STAMP" >&2; exit 2;; esac
case "$DATASET" in *[!a-zA-Z0-9_.-]*|'') echo "Invalid DATASET=$DATASET" >&2; exit 2;; esac
[[ -f "$ROOT/TCRR-STLA/pbs/formal_v5.pbs" ]] || { echo "Missing formal_v5.pbs" >&2; exit 2; }
[[ -f "$BACKBONE_CHECKPOINT" ]] || { echo "Missing Backbone: $BACKBONE_CHECKPOINT" >&2; exit 2; }

MODE_DIR=formal
if [[ "$SMOKE" == 1 ]]; then MODE_DIR=smoke; fi
RUN_DIR="$ROOT/TCRR-STLA_exps/$TARGET/$BACKBONE_MODE/$DATASET/v5/$MODE_DIR/${RUN_STAMP}_seed${SEED}_full"
[[ ! -e "$RUN_DIR" ]] || { echo "Run exists: $RUN_DIR" >&2; exit 2; }
VARS="TARGET=$TARGET,SEED=$SEED,DATASET=$DATASET,BACKBONE_MODE=$BACKBONE_MODE,BACKBONE_CHECKPOINT=$BACKBONE_CHECKPOINT,TFP_ENV_ROOT=$TFP_ENV_ROOT,RUN_STAMP=$RUN_STAMP,RUN_DIR=$RUN_DIR"
# Keep every documented training/evaluation override in the submitted PBS environment.
for key in BATCH_SIZE TCRR_TRAIN_WORKERS TCRR_VAL_WORKERS TCRR_EVAL_WORKERS TCRR_PERSISTENT_WORKERS TCRR_PREFETCH_FACTOR DIAGNOSTIC_BATCH_SIZE DIAGNOSTIC_MAX_BATCHES DIAGNOSTIC_PRIMARY_MAX_BATCHES; do
  if [[ -n "${!key:-}" ]]; then VARS+=",$key=${!key}"; fi
done

# PBS -l overrides static #PBS directives in the target scripts.
PBS_RESOURCES=(-l "ncpus=$TCRR_PBS_NCPUS" -l "mem=$TCRR_PBS_MEM")

if [[ "$SMOKE" == 1 ]]; then
  # Smoke is NOT a formal experiment; never schedule dependent diagnostics/test.
  SMOKE_VARS="$VARS,SMOKE=1,SMOKE_PROFILE_FULL_BATCH=$SMOKE_PROFILE_FULL_BATCH,SKIP_DIAGNOSTICS=1,SKIP_EVALUATE=1"
  SMOKE_ID=$(qsub -N "T53_${TARGET}_sm" "${PBS_RESOURCES[@]}" -l walltime=04:00:00 -v "$SMOKE_VARS" TCRR-STLA/pbs/formal_v5.pbs)
  printf 'SMOKE_ID=%s\nRUN_DIR=%s\nSMOKE_PROFILE_FULL_BATCH=%s\nCONFIGURED_BATCH_SIZE=%s\nEFFECTIVE_SMOKE_BATCH_SIZE=%s\n' \
    "$SMOKE_ID" "$RUN_DIR" "$SMOKE_PROFILE_FULL_BATCH" "$BATCH_SIZE" "$( [[ "$SMOKE_PROFILE_FULL_BATCH" == 1 ]] && echo "$BATCH_SIZE" || echo 3 )"
  exit 0
fi
# TRAIN uses the accelerated 96GB (overridable) profile.
# POST is a single GPU job: validation diagnostics MUST pass before locked test.
TRAIN_ID=$(qsub -N "T53_${TARGET}_tr" "${PBS_RESOURCES[@]}" -v "$VARS,SKIP_DIAGNOSTICS=1,SKIP_EVALUATE=1" TCRR-STLA/pbs/formal_v5.pbs)
POST_RESOURCES=(-l "ncpus=$TCRR_PBS_NCPUS" -l "mem=$TCRR_POST_PBS_MEM" -l "walltime=$TCRR_POST_WALLTIME")
if ! POST_ID=$(qsub -N "T53_${TARGET}_pp" "${POST_RESOURCES[@]}" -W "depend=afterok:$TRAIN_ID" -v "$VARS" TCRR-STLA/pbs/postprocess_v5.pbs); then
  echo "ERROR: postprocessing submission failed, but TRAIN_ID=$TRAIN_ID was already submitted. Inspect/cancel it explicitly if needed." >&2
  exit 1
fi
printf 'TRAIN_ID=%s\nPOSTPROCESS_ID=%s\nRUN_DIR=%s\nBATCH_SIZE=%s\nPBS_NCPUS=%s\nTRAIN_PBS_MEM=%s\nPOST_PBS_MEM=%s\nPOST_WALLTIME=%s\nTRAIN_WORKERS=%s\n' \
  "$TRAIN_ID" "$POST_ID" "$RUN_DIR" "$BATCH_SIZE" "$TCRR_PBS_NCPUS" \
  "$TCRR_PBS_MEM" "$TCRR_POST_PBS_MEM" "$TCRR_POST_WALLTIME" "$TCRR_TRAIN_WORKERS"
