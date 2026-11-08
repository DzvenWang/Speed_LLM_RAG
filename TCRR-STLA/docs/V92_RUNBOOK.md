# TCRR-STLA v9.2.0 — Gadi deployment and reproducible diagnosis

**Requirements**: copy the full source ZIP into `/scratch/mh20/zw1165/Speed_LLM_RAG/` so that
`TCRR-STLA/`, `Backbone/` and `Base_SupportBench/` are siblings. Preserve the original
STLAformer best.pt and v5.3 source protocol. All environments, logs, cache roots
and experiment files are independent of v9.1. Do NOT resume any v9.1 checkpoint.

## Formal jobs (NO GPU smoke required)

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP
export STLA_SPEED="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_FLOW="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
export V92_SPEED_RUN="$PWD/TCRR-STLA_exps/speed/TraffiDent_Net_06/v9_2/formal/run_seed10"
export V92_FLOW_RUN="$PWD/TCRR-STLA_exps/flow/TraffiDent_Net_06/v9_2/formal/run_seed10"
test -f TCRR-STLA/scripts/train_v92.py && test -f TCRR-STLA/pbs/formal_v92.pbs
qsub -N TCRR92_speed \
  -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$STLA_SPEED",V92_STAGE=all,V92_RUN_DIR="$V92_SPEED_RUN",TFP_ENV_ROOT="$TFP_ENV_ROOT" \
  TCRR-STLA/pbs/formal_v92.pbs
qsub -N TCRR92_flow \
  -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$STLA_FLOW",V92_STAGE=all,V92_RUN_DIR="$V92_FLOW_RUN",TFP_ENV_ROOT="$TFP_ENV_ROOT" \
  TCRR-STLA/pbs/formal_v92.pbs
qstat -u "$USER"
```

PBS uses `gpuvolta` with 1 V100, 12 CPUs, 96 GiB host RAM and a 48h
walltime. It runs formal preflight, source-locked adapter, strong original
retrieval on Official view, 2-view historical replay, training stages and
post-training diagnostics. The proper GPU requirement is a preflight check,
**not** a separate GPU smoke training run. CPU synthetic tests are in `tests/v92`.

Real Physical Upstream/Downstream requires **both**:
`V92_PHYSICAL_GRAPH=/absolute/path/graph.npz,V92_NODE_IDS=/absolute/path/ids.npy`.
Real regional/directional outages additionally require
`V92_REGIONS=/absolute/path/physically_grounded_regions.npy`.
Otherwise, physical GNN branches are inactive and these mask families fall back
explicitly to sensor/temporal outages.

## Logs, recovery and monitoring

```bash
ls -t TCRR-STLA_logs/speed/v9_2/*.live.log | head
ls -t TCRR-STLA_logs/flow/v9_2/*.live.log | head
```

Watch markers: `V92_PREFLIGHT_PASS`, `V92_PREPARATION_PASS`, `V92_EPOCH`,
`V92_TRAIN_PASS`, `V92_DIAGNOSTIC_PASS`, `V92_AUTO_DIAGNOSTICS_PASS`, `V92_PBS_FINISH`.
Each stage saves `best_normal.pt`, `best_align.pt`, `best_utility.pt`,
`best_joint.pt`, `best_paired.pt` (if that stage runs) and `last.pt`.
`selected_checkpoint.json` points to the stage checkpoint selected solely on
chronological Validation Selection. `history.json` includes gradients, CF Gain,
Null activation, alignment use, GPU peak and paired matching rate.

To resume, use `V92_STAGE=train,V92_RESUME=/abs/run/last.pt,V92_RUN_DIR=/abs/run`.
All training options must match the saved run exactly. Normal, stage and optimizer
RNG metadata are checked; do not substitute a v9.1 checkpoint.

## Automatic post-training analysis

By default it runs on Validation Chronological Check (256 queries, unless fewer),
with a separate same-cohort Wrong-mask Replay control (32 queries) and an
optimistic retrospective Headroom grid. Open
`$V92_SPEED_RUN/diagnostics/report.html` and `summary.json`.
Reports distinguish behavior-only checkpoint controls from separately retrained
ablation evidence. Failure groups use **observed missing fraction bins**, not
invented s20/s40/s60/s90 labels on Official Validation view.

To redo diagnostics:

```bash
python -u TCRR-STLA/scripts/diagnose_v92.py \
  --run-identity "$V92_SPEED_RUN/run_identity.json" \
  --max-queries 256 --wrong-mask-queries 32
```

The standalone full-control Replay requires the original STLA checkpoint and
adapter, which the run identity verifies. It consumes GPU resources.

## Blind severity TEST after training/diagnosis and model selection

```bash
SELECTED=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["path"])' "$V92_SPEED_RUN/selected_checkpoint.json")
qsub -N TCRR92_evalS \
  -v TARGET=speed,BACKBONE_CHECKPOINT="$STLA_SPEED",V92_STAGE=evaluate,V92_RUN_IDENTITY="$V92_SPEED_RUN/run_identity.json",V92_CHECKPOINT="$SELECTED",TFP_ENV_ROOT="$TFP_ENV_ROOT" \
  TCRR-STLA/pbs/formal_v92.pbs
```

Repeat for Flow by replacing the checkpoint and run identity. Evaluation writes
`test_s20.json` to `test_s90.json` once and **refuses overwrite**. TEST outcomes
are never read during selection, teacher replay or CF inference.

## Independent mechanism ablations

Use **separate** `V92_RUN_DIR`s with otherwise identical conditions:

- No learnable CF Graph: `V92_GRAPH_MODE=none`
- Fixed graph aggregation: `V92_GRAPH_MODE=fixed`
- GNN without reliability gating: `V92_GRAPH_MODE=gnn`
- Positive-only alignment: `V92_ALIGNMENT_MODE=positive`
- Raw alignment: `V92_ALIGNMENT_MODE=raw`
- Uniform candidate scores: `V92_UTILITY_MODE=uniform`
- Similarity-only candidate scores: `V92_UTILITY_MODE=similarity`
- Fixed Null: `V92_NULL_MODE=fixed`
- Same-budget paired augmentation WITHOUT utility-difference and compatibility losses:
  `V92_PAIRED_LOSS_OFF=1`, **keeping** `PAIRED_EPOCHS` the same.
- No paired stage: `PAIRED_EPOCHS=0`; note this differs in training budget.

After independent retraining, compare same-normal and same-cohort runs:

```bash
python -u TCRR-STLA/scripts/compare_ablation_v92.py \
  /abs/path/full_run /abs/path/no_graph_run /abs/path/positive_run \
  --output /abs/path/ablation_comparison.json
```

The comparison script rejects runs with differing Normal parameter signatures,
Backbone SHA256, evidence root, validation cohort, seed, target or dataset.

## Boundaries of verification

Included package has passed Python compilation, PBS Bash syntax, v9.1 targeted
regression and v9.2 CPU synthetic full-stage preparation/training/diagnostics.
The complete legacy test suite takes more than the available execution window
and must not be reported as fully passed. No live Gadi V100 run or full-scale
speed/flow benchmark has yet been executed; monitor real memory and runtime.
