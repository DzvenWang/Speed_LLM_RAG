#!/usr/bin/env bash
# Usage: MODE=all|diagnostic|test|verify|aggregate bash ... speed|flow /abs/path/best.pt [seed]
# MODE=all: ONE frozen selection diagnostic followed by TEST s20/s40/s60/s90.
# QUEUE=cpu (default) or QUEUE=gpu (test mode only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
TARGET="${1:?speed or flow}"
CHECKPOINT="${2:?Absolute original STLAformer best.pt}"
SEED="${3:-10}"
[[ "$TARGET" == speed || "$TARGET" == flow ]] || { echo 'speed|flow only' >&2; exit 2; }
[[ "$CHECKPOINT" == /* && -f "$CHECKPOINT" ]] || { echo 'Absolute checkpoint required' >&2; exit 2; }
MODE="${MODE:-all}"
QUEUE="${QUEUE:-cpu}"
[[ "$MODE" =~ ^(all|diagnostic|test|verify|aggregate)$ ]] || exit 2
[[ "$QUEUE" == cpu || "$QUEUE" == gpu ]] || exit 2
[[ "$QUEUE" == cpu || "$MODE" == test ]] || { echo 'GPU alternative is TEST-only; run diagnostic on CPU.' >&2; exit 2; }
DATASET="${DATASET:-TraffiDent_Net_06}"
BACKBONE_MODE="${BACKBONE_MODE:-incomplete}"
export TFP_ENV_ROOT="${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}"
PYTHON="$TFP_ENV_ROOT/bin/python"
[[ -x "$PYTHON" ]] || { echo "TFP interpreter missing $PYTHON" >&2; exit 2; }
export SPEED_LLM_RAG_ROOT="$ROOT" BACKBONE_MODE TRAFFIDENT_TARGET="$TARGET"
export PYTHONPATH="$ROOT/TCRR-STLA:$ROOT/Backbone:$ROOT/Base_SupportBench:${PYTHONPATH:-}"
RUN_DIR="${RUN_DIR:-$ROOT/TCRR-STLA_exps/$TARGET/$BACKBONE_MODE/$DATASET/v7_1_0_c1/formal/seed$SEED}"
[[ -e "$RUN_DIR/run_identity.json" ]] || { echo 'Run identity not found; use completed v7.1 run' >&2; exit 2; }
# No source file included in the original v7.1 code digest is changed by this add-on.
"$PYTHON" -m pytest -q TCRR-STLA/tests
"$PYTHON" -u TCRR-STLA/scripts/c1_frozen_analysis.py --mode verify \
  --target "$TARGET" --dataset "$DATASET" --seed "$SEED" \
  --backbone-checkpoint "$CHECKPOINT" --run-dir "$RUN_DIR" --device cpu --threads 4
ANALYSIS_SHA256="$("$PYTHON" -c 'import hashlib,pathlib; p=pathlib.Path("TCRR-STLA/scripts/c1_frozen_analysis.py"); print(hashlib.sha256(p.read_bytes()).hexdigest())')"
RES="TARGET=$TARGET,DATASET=$DATASET,SEED=$SEED,BACKBONE_MODE=$BACKBONE_MODE,BACKBONE_CHECKPOINT=$CHECKPOINT,TFP_ENV_ROOT=$TFP_ENV_ROOT,SPEED_LLM_RAG_ROOT=$ROOT,RUN_DIR=$RUN_DIR,ANALYSIS_SHA256=$ANALYSIS_SHA256,MODE=$MODE,EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8},CPU_THREADS=${CPU_THREADS:-8},BOOTSTRAP=${BOOTSTRAP:-400},BLOCK_STEPS=${BLOCK_STEPS:-288},SEVERITIES=${SEVERITIES:-20:40:60:90},CACHE_ONLY=${CACHE_ONLY:-0},NO_ORIGINAL_STLA=${NO_ORIGINAL_STLA:-0}"
JOBFILE='TCRR-STLA/pbs/c1_frozen_cpu.pbs'
[[ "$QUEUE" == gpu ]] && JOBFILE='TCRR-STLA/pbs/c1_frozen_gpu_test.pbs'
ID="$(qsub -N "C1_${MODE}_${TARGET}" -v "$RES" "$JOBFILE")"
echo "C1_FROZEN_JOB_ID=$ID"
echo "C1_FROZEN_TARGET=$TARGET MODE=$MODE QUEUE=$QUEUE"
echo "C1_FROZEN_RUN_DIR=$RUN_DIR"
echo "C1_FROZEN_SOURCE_SHA=$ANALYSIS_SHA256"
