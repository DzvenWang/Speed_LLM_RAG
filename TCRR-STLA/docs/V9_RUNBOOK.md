# TCRR-STLA v9.1.0 — Gadi runbook

## Setup

From the existing project root (contains `Backbone`, `Base_SupportBench`, and an installed CUDA environment `TFP`):

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export TFP_ENV_ROOT=/g/data/mh20/zw1165/envs/TFP
export STLA_SPEED="$PWD/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_FLOW="$PWD/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
```

**Replace the existing `TCRR-STLA` source directory with the v9.1 package only after preserving a copy of your previous code.** The root legacy `VERSION` stays at v5.3 for historical tests; the active v9.1 identity lives in `tcrr_stla/v9/VERSION`, the YAML and each checkpoint. v9.0.0 checkpoints must NOT be used as resume checkpoints for v9.1.

## Formal submission, without GPU Smoke

Direct preparation and staged training in one PBS job:

```bash
qsub -N TCRR91_Speed \
 -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$STLA_SPEED",V9_STAGE=all,V9_RUN_DIR="$PWD/TCRR-STLA_exps/speed/TraffiDent_Net_06/v9_1/formal/run_seed10",TFP_ENV_ROOT="$TFP_ENV_ROOT" \
 TCRR-STLA/pbs/formal_v9.pbs

qsub -N TCRR91_Flow \
 -v TARGET=flow,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$STLA_FLOW",V9_STAGE=all,V9_RUN_DIR="$PWD/TCRR-STLA_exps/flow/TraffiDent_Net_06/v9_1/formal/run_seed10",TFP_ENV_ROOT="$TFP_ENV_ROOT" \
 TCRR-STLA/pbs/formal_v9.pbs
```

For repeatable experiments, prefer separate `V9_STAGE=prepare` then `V9_STAGE=train`, after the preparation stage prints `V9_PREPARATION_PASS`. Both stages use the same target/seed/checkpoint and the new `v9_1_isolated` evidence namespace.

By default Normal 8 epochs → Transfer 15 → Utility 15 → Cross-Failure 4. Override with `NORMAL_EPOCHS`, `TRANSFER_EPOCHS`, `UTILITY_EPOCHS`, `CROSS_EPOCHS` (use `0` to ablate Cross). Default batch 32, 4 DataLoader workers, preparator batch 4. A trusted upstream/downstream physical graph can be supplied with `V9_PHYSICAL_GRAPH`, `V9_NODE_IDS` and region labels with `V9_REGIONS`.

## Resume

`last.pt` is atomic and stores stage/epoch, optimizer, scaler, RNG. Use same run dir, settings and v9.1 code:

```bash
qsub -N TCRR91_ResumeSpeed \
 -v TARGET=speed,DATASET=TraffiDent_Net_06,SEED=10,BACKBONE_CHECKPOINT="$STLA_SPEED",V9_STAGE=train,V9_RUN_DIR="$PWD/TCRR-STLA_exps/speed/TraffiDent_Net_06/v9_1/formal/run_seed10",V9_RESUME="$PWD/TCRR-STLA_exps/speed/TraffiDent_Net_06/v9_1/formal/run_seed10/last.pt",TFP_ENV_ROOT="$TFP_ENV_ROOT" \
 TCRR-STLA/pbs/formal_v9.pbs
```

## Evaluation and diagnostics

Run Validation-Selection diagnostic before locking a test checkpoint. The standard wrong-mask negative control is expensive because it replays the historical teacher and limits **all** comparison modes to the same 32 Query rows. Use `--wrong-mask-max 0` for a larger no-wrong-mask comparison.

```bash
python -u TCRR-STLA/scripts/diagnose_v9.py \
 --checkpoint TCRR-STLA_exps/speed/TraffiDent_Net_06/v9_1/formal/run_seed10/best_cross.pt \
 --run-identity TCRR-STLA_exps/speed/TraffiDent_Net_06/v9_1/formal/run_seed10/run_identity.json \
 --max-queries 256 --batch-size 8 --wrong-mask-max 32
```

For final, locked four-severity TEST, set `V9_STAGE=evaluate`, `V9_CHECKPOINT`, `V9_RUN_IDENTITY` and `TARGET` in `pbs/formal_v9.pbs`. The script refuses to overwrite existing `test_s20/s40/s60/s90.json` files. Default physical graph absent means **Functional only**; absent physical regions cause `regional_outage → sensor_outage` and `directional_outage → temporal_outage`, with actual family counts recorded.

## Resource/checks

- CUDA V100 32 GiB, 12 CPUs, 96 GB CPU RAM, 48h PBS. These resources are starting defaults, not an empirically established optimum.
- `source TFP` via PBS `pbs/runtime.sh`; `V9_PREPARATION_PASS`, `V9_EPOCH` and `V9_TRAIN_PASS` indicate stage completion. `V9_EPOCH` Cross stage logs `matching_rate` and candidate counts.
- Both paired views have identical historical candidate IDs by design, and different current masks. No hidden full Query data enters the prediction-time model.
- **Version isolation:** new evidence `v9_1_isolated`, experiment `v9_1`, checkpoint version `9.1.0`. Old v9.0.0 checkpoints are incompatible; v9.0 replay must not be reused as v9.1 evidence.
- Targeted CPU tests: `python -m pytest -q TCRR-STLA/tests/v9`; original v5 tests remain bundled for longer regression checks.

**Status:** Gadi V100/real-data execution has not been performed in this environment. Real-run outcomes and MAE gains remain unknown.
