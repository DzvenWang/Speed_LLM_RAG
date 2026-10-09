# Reference Backbone: incomplete / complete selection (Baseline v1.4.5)

The Baseline *query inputs stay incomplete* under the same Base_SupportBench Core-5 contract. `BACKBONE_TRAIN_MODE` selects only **which trained, frozen Backbone checkpoint** is used to generate reference forecasts or to forecast restored inputs. All Baseline experiments continue to test on `s20`, `s40`, `s60`, and `s90`.

## Choosing a checkpoint

| `BACKBONE_TRAIN_MODE` | Backbone source | Baseline formal runs (Backbone-dependent methods) |
| --- | --- | --- |
| `incomplete` (default) | `Backbone_exps/{target}/{model}/{dataset}/formal/` | `Baseline_exps/{target}/{method}/{model}/{dataset}/formal/` |
| `complete` | `Backbone_exps/{target}/{model}/{dataset}/formal_complete/` | `Baseline_exps/{target}/{method}/{model}/complete/{dataset}/formal/` |

Both modes select a seed-matched `best.pt`. The saved Backbone checkpoint's `train_data_mode` is checked explicitly. Missing mode fields on historical checkpoint files mean `incomplete`. A `complete` request NEVER silently falls back to `incomplete`. If `BASELINE_BACKBONE_CHECKPOINT` or `--backbone-checkpoint` is set, it must also match the selected mode, model, target, seed, and formal shared cache identity.

## Gadi PBS runs

Run from `Speed_LLM_RAG` after the relevant Backbone `complete` training finishes and produces `best.pt`.

```bash
# Frozen complete-trained STLAformer + baseline RAFT correction, speed
qsub -N RAFT_full_speed \
  -v METHOD=RAFT-STLAformer,BACKBONE=STLAformer,TARGET=speed,SEED=10,BACKBONE_TRAIN_MODE=complete \
  Baseline/pbs/methods/formal_primary.pbs

# Same Baseline on a complete-trained STLAformer, flow
qsub -N RAFT_full_flow \
  -v METHOD=RAFT-STLAformer,BACKBONE=STLAformer,TARGET=flow,SEED=10,BACKBONE_TRAIN_MODE=complete \
  Baseline/pbs/methods/formal_primary.pbs

# Original incomplete-trained STLAformer remains the default
qsub -N RAFT_inc_speed \
  -v METHOD=RAFT-STLAformer,BACKBONE=STLAformer,TARGET=speed,SEED=10 \
  Baseline/pbs/methods/formal_primary.pbs
```

For `RAST-STLAformer`, `ImputeFormer`, and `LRTC-TMCP`, change `METHOD` while keeping `BACKBONE=STLAformer`. The `RAFT-PatchSTG`/`RAST-PatchSTG` cases also support mode selection when that Backbone has a matching checkpoint. Independent methods (e.g., native `RAFT`, `RAST`, `GinAR`, `ReMFT`) do not use a frozen reference Backbone; PBS explicitly rejects `complete` for them to prevent misleading comparisons.

The PBS script preflights the corresponding Backbone checkpoint, runs the Baseline, and automatically evaluates all four severities unless `AUTO_EVAL=0`.

## Local / manual commands

```bash
# Inspect available reference Backbone checkpoint for one mode
export SPEED_LLM_RAG_ROOT="$PWD"
export TRAFFIDENT_TARGET=speed
export BACKBONE_TRAIN_MODE=complete
python -u Baseline/scripts/preflight_formal.py --method RAFT-STLAformer \
  --stage formal --seed 10 --backbone STLAformer --backbone-train-mode complete

# Reevaluate the latest completed matching Baseline Formal run (from repository root)
python -u Baseline/scripts/run_evaluations.py --method RAFT-STLAformer \
  --seed 10 --backbone STLAformer --backbone-train-mode complete

# OR select an exact matching completed Baseline formal run
python -u Baseline/scripts/run_evaluations.py --method RAFT-STLAformer \
  --seed 10 --backbone STLAformer --backbone-train-mode complete \
  --formal-run-dir '/path/to/Baseline_exps/speed/RAFT-STLAformer/STLAformer/complete/TraffiDent_Net_06/formal/<run>_seed10'
```

You may also add `--backbone-train-mode complete|incomplete` to the supported preflight, runner, and evaluation commands without setting the environment variable. When both are given, the CLI argument takes precedence **within that invocation**.

## Evaluation outputs

`complete` results live at `Baseline_exps/{target}/evaluation/{method}/{model}/complete/{dataset}/seed10/` with per-severity subfolders; old `incomplete` evaluation outputs stay in `{method}/{model}/{dataset}/seed10/`.

`summary.json`, `best.pt`, and evaluation summaries record `reference_backbone_train_mode`, along with the reference checkpoint path and SHA256 for reproducibility. Existing Baseline formal checkpoints with no mode field are treated as incomplete.

## Regression tests

```bash
cd Baseline
python -m unittest discover -s tests -v
bash -n pbs/methods/formal_primary.pbs
```

**Scope**: v1.4.5 changes only the reference-Backbone checkpoint selection, provenance, and per-mode output isolation. It does not change the Baseline reconstruction/retrieval algorithm, frozen model weights, cache sampling, or test missingness. CPU-only unit tests passed; Gadi GPU formal runs have not been exercised in this package.
