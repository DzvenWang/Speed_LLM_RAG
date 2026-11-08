# v9.3 Runbook — Gadi

1. Copy the complete ZIP's `TCRR-STLA/` folder into `$SPEED_LLM_RAG_ROOT/TCRR-STLA/` (preserving other project roots, `Backbone` and `Base_SupportBench`).
2. Confirm source Base data and Backbone cache exist; do not overwrite Base SupportBench.
3. Activate TFP and use the same STLA checkpoint + trained adapter for all reference and masked teacher passes.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export STLA_SPEED="$(pwd)/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
test -f "$STLA_SPEED"
python -u TCRR-STLA/scripts/preflight_v93.py --target speed --dataset TraffiDent_Net_06 --backbone-checkpoint "$STLA_SPEED" --require-gpu
```

### Step A: Prepare multi-mask teacher (GPU)

```bash
qsub -N V93_SPD_PREP \
  -v TARGET=speed,SEED=10,V93_STAGE=prepare,BACKBONE_CHECKPOINT="$STLA_SPEED" \
  TCRR-STLA/pbs/formal_v93.pbs
```

This trains/reuses the versioned v5 S0 mask adapter, generates official + multi-mask TRAIN teachers, and caches only official VAL predictions/contexts. With 50k+ windows and 4 views, preparation may take substantial computation; it is resumable by its `progress.json` and immutable identity.

### Step B: Train (GPU)

```bash
qsub -N V93_SPD_TRAIN \
  -v TARGET=speed,SEED=10,V93_STAGE=train,BACKBONE_CHECKPOINT="$STLA_SPEED",V93_MODE=incremental,V93_GRAPH_MODE=none \
  TCRR-STLA/pbs/formal_v93.pbs
```

Pass the exact same checkpoint, target, dataset, mask plan and source versions. Adapter and teacher root are deterministic from their hashes. Logs: `TCRR-STLA_logs/speed/v9_3/`, model runs: `TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v9_3/` (exact path depends on v5 cfg).

After the job starts, the run folder includes `run_identity.json`, `best_residual.pt`, `best_cf_response.pt`, `best_utility.pt`, `best_cf_joint.pt`, `calibration.json`, `check_report.json` and `diagnostics/`.

### Step C: Evaluate (locked TEST)

Use absolute paths resolved from the actual completed TRAIN run. The four severities run independently with `Backbone`'s `evaluation` cache. **Each TEST result file is write-once**.

```bash
qsub -N V93_SPD_TEST \
  -v TARGET=speed,DATASET=TraffiDent_Net_06,V93_RUN_DIR=/absolute/path/to/completed/run,V93_CHECKPOINT=/absolute/path/to/completed/run/best_cf_joint.pt,BACKBONE_CHECKPOINT="$STLA_SPEED",V93_ADAPTER_CHECKPOINT=/absolute/path/to/v93_adapter/adapter.pt \
  TCRR-STLA/pbs/evaluate_v93.pbs
```

`test_s20.json`, `test_s40.json`, `test_s60.json`, `test_s90.json` report raw MAE/RMSE, masked MAE, H12, Residual and incremental CF gain, harm and activation. Nothing consults full TEST reference data at inference.

### Comparative modes

* `V93_MODE=incremental`: strongest masked Residual then CF utility.
* `V93_MODE=decomposed`: reference Residual plus supervised CF.
* `V93_MODE=single`: Residual-only structural control; CF disabled.
* `V93_GRAPH_MODE=none|fixed|v7_edge_attention`: independent training runs, not toggles within same checkpoint.
* Preserve `configs/tcrr_stla_v92.yaml` / old v9.2 runs as retrieval-based baseline; do not mix v92 cache with v93.

Use separate `V93_RUN_DIR` per experiment, and keep source/data/checkpoint identity unchanged during resume. `--resume /path/to/last.pt --output /path/to/run` resumes the stage. For a selected other run, rerun preparation only if Teacher identity differs; graph experiments reuse the same Teacher evidence.

### Real-hardware caution

The release was tested with CPU synthetic traffic and official mask generator; it has not run genuine Gadi hardware or datasets. Run preflight, confirm disk availability for the Memmaps, then monitor `V93_TEACHER_PROGRESS`, `V93_EPOCH`, `V93_DIAGNOSTIC_PASS`, GPU memory and batch throughput.
