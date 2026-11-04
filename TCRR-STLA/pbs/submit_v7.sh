#!/usr/bin/env bash
# Usage: bash TCRR-STLA/pbs/submit_v7.sh speed|flow /absolute/path/to/best.pt [seed]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
TARGET="${1:?speed|flow}";CKPT="${2:?absolute checkpoint best.pt}";SEED="${3:-10}"
[[ "$TARGET" == speed || "$TARGET" == flow ]] || exit 2
[[ "$CKPT" == /* && -f "$CKPT" && "${CKPT##*/}" == best.pt ]] || { echo 'Absolute original Backbone best.pt required' >&2;exit 2; }
DATASET="${DATASET:-TraffiDent_Net_06}"
BACKBONE_MODE="${BACKBONE_MODE:-incomplete}"
case "$CKPT" in
 "$ROOT/Backbone_exps/$TARGET/STLAformer/$DATASET/formal/"*) [[ "$BACKBONE_MODE" == incomplete ]] || exit 2 ;;
 "$ROOT/Backbone_exps/$TARGET/STLAformer/$DATASET/formal_complete/"*) [[ "$BACKBONE_MODE" == complete ]] || exit 2 ;;
 *) echo 'Backbone checkpoint target/mode mismatch' >&2;exit 2 ;;
esac
export TFP_ENV_ROOT="${TFP_ENV_ROOT:-/g/data/mh20/zw1165/envs/TFP}"
export SPEED_LLM_RAG_ROOT="$ROOT" PYTHONPATH="$ROOT/TCRR-STLA:$ROOT/Backbone:$ROOT/Base_SupportBench:${PYTHONPATH:-}" BACKBONE_MODE
PYTHON="$TFP_ENV_ROOT/bin/python"
"$PYTHON" -m pytest -q TCRR-STLA/tests
META=$("$PYTHON" -c 'import sys,torch;c=torch.load(sys.argv[1],map_location="cpu",weights_only=False);m=c["model_config"];print(int(c["num_nodes"]),int(m.get("input_embedding_dim",32))+int(m.get("tow_embedding_dim",32)))' "$CKPT")
read -r PREFLIGHT_NODES PREFLIGHT_MODEL_DIM <<< "$META"
C1_META=$("$PYTHON" -c 'import sys,yaml;c=yaml.safe_load(open(sys.argv[1]));print(int(c["model"]["top_k"]),int(c["model"]["hidden"]))' TCRR-STLA/configs/tcrr_stla_v7.yaml)
read -r PREFLIGHT_K PREFLIGHT_HIDDEN <<< "$C1_META"
"$PYTHON" -u TCRR-STLA/scripts/preflight_c1_v71.py --device cpu \
  --nodes "$PREFLIGHT_NODES" --model-dim "$PREFLIGHT_MODEL_DIM" \
  --top-k "$PREFLIGHT_K" --hidden "$PREFLIGHT_HIDDEN" --batch-size 2
"$PYTHON" -u TCRR-STLA/scripts/preflight_v7.py --target "$TARGET" --dataset "$DATASET" --backbone-checkpoint "$CKPT"
RUN_DIR="$ROOT/TCRR-STLA_exps/$TARGET/$BACKBONE_MODE/$DATASET/v7_1_0_c1/formal/seed$SEED"
: "${BATCH_SIZE:=16}"
: "${TRAIN_WORKERS:=4}"
# Catch parent/child identity mismatches before GPU queueing.
# The check is intentionally called twice: first creates the identity, second
# validates that JSON serialization did not change the expected object.
for pass in 1 2; do
  "$PYTHON" -u TCRR-STLA/scripts/run_v7.py --target "$TARGET" --dataset "$DATASET" \
    --seed "$SEED" --backbone-checkpoint "$CKPT" --run-dir "$RUN_DIR" \
    --batch-size "$BATCH_SIZE" --workers "$TRAIN_WORKERS" --end-stage c1 --check-run-identity
done
# Optional old cache reuse: require explicit source; import is fail-closed and only
# marks S0/cache/replay completed after validating every chunk hash.
if [[ -n "${REUSE_V7_RUN_DIR:-}" ]]; then
  "$PYTHON" -u TCRR-STLA/scripts/reuse_v7_caches.py \
     --source "$REUSE_V7_RUN_DIR" --destination "$RUN_DIR" \
     --target "$TARGET" --dataset "$DATASET" --seed "$SEED" \
     --backbone-checkpoint "$CKPT"
fi
ID=$(qsub -N "T7_${TARGET}" -v "TARGET=$TARGET,DATASET=$DATASET,SEED=$SEED,BACKBONE_MODE=$BACKBONE_MODE,BACKBONE_CHECKPOINT=$CKPT,TFP_ENV_ROOT=$TFP_ENV_ROOT,SPEED_LLM_RAG_ROOT=$ROOT,RUN_DIR=$RUN_DIR,BATCH_SIZE=$BATCH_SIZE,TRAIN_WORKERS=$TRAIN_WORKERS" TCRR-STLA/pbs/formal_v7.pbs)
echo "V7_JOB_ID=$ID"
echo "V7_RUN_DIR=$RUN_DIR"
