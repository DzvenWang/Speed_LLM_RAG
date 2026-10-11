# FARE v2.0.2 — Pilot first (Speed/Flow)

This adds a dedicated PBS job and updates the pilot YAML to match all v2.0.2 features. The original `full.yaml` and `train_formal.pbs` are unchanged.

## Protocol

- TraffiDent_Net_06 by default; seed 10; Speed and Flow in separate allocations
- 8,000 evenly spread training windows; full validation split; 16,000 TRAIN-only memory windows
- 5 training phases: retrieval=5, recovery=5, injection=10, partial=5, full=5 (max 30 epochs)
- Batch Size 32, 2 loader workers; 1 V100, 12 CPUs, 32 GB RAM, 24-hour allocation
- Forecast Utility, retrieval Top-64/Top-8, diagnostics, hardware monitoring retained
- In-job `val` evaluation after training, **not test**; no formal-performance claims from this pilot

## Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
python -u FARE/scripts/preflight.py --target speed --config FARE/configs/pilot.yaml
python -u FARE/scripts/preflight.py --target flow --config FARE/configs/pilot.yaml
qsub -N FARE_PILOT_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
qsub -N FARE_PILOT_flow -v TARGET=flow,SEED=10,BATCH_SIZE=32 FARE/pbs/train_pilot.pbs
```

For a reduced-resource test, override BATCH_SIZE=16. To change monitoring, specify FARE_RESOURCE_INTERVAL=60 or FARE_DIAG_INTERVAL=100 in qsub `-v`.

Outputs: `FARE_exps/<target>/<dataset>/pilot/<stamp>_seed10/` and `FARE_logs/<target>/<dataset>/...`. Diagnostics in the run directory and resource JSONL/CSV by the log.

**Caution**: The pilot is a separate training run, not a short `--smoke` and not a resumable continuation of a different full.yaml run. Avoid changing the underlying FARE directory while other jobs are running.
