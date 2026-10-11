# FARE v2.0.2 — Monitoring & Intermediate Diagnostics

This archive is the **full, standalone FARE project**. It preserves v2.0.1's training-speed fixes, STLAformer, Forecast Utility ranking, five-phase training, train-only memory, PBS scripts, checkpoint resume, and post-training automatic evaluation.

## Install & preflight

Back up the existing `FARE/` directory. Install this archive as `Speed_LLM_RAG/FARE/` before starting **new** PBS jobs. Do not replace code underneath a currently running Gadi job.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
python -u FARE/scripts/preflight.py --target speed
python -u FARE/scripts/preflight.py --target flow
python -u FARE/scripts/train.py --target speed --seed 10 --smoke
```

## Formal training with batch 32

```bash
qsub -N FARE_v202_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32 FARE/pbs/train_formal.pbs
qsub -N FARE_v202_flow  -v TARGET=flow,SEED=10,BATCH_SIZE=32 FARE/pbs/train_formal.pbs
```

Each PBS job records two kinds of diagnostics:

1. **Independent sampled resource usage** (GPU util/VRAM/power/temperature, job CPU cores, RSS), saved as `FARE_logs/<target>/<dataset>/*.resources.{jsonl,csv,summary.json}`.
2. **Model development diagnostics**, saved to the generated run directory `FARE_exps/.../<run>/diagnostics/`: `epoch_diagnostics.csv`, `epoch_diagnostics.jsonl`, `epoch_resource.csv`, `telemetry_report.md`, `telemetry_report.json`.

The stdout training log includes `FARE_DIAG`, `FARE_PROGRESS`, `FARE_RESOURCE_EPOCH`, and existing `FARE_EPOCH`/`FARE_STAGE_DIAG` events.

## Controlling cost

- `FARE_RESOURCE_INTERVAL=30` (seconds) — GPU/CPU sampler, independent of training.
- `FARE_DIAG_INTERVAL=50` (batches) — diagnostics sampled on existing batches.
- `FARE_DIAG_MAX_BATCHES=160` — maximum diagnostic batches per epoch.
- `FARE_PROGRESS_EVERY=500` — log heartbeat interval.
- `FARE_DISABLE_DIAGNOSTICS=1` — skip model diagnostics, retains hardware sampler.
- `FARE_RESOURCE_MONITOR=0` — skip hardware sampler, retains model diagnostics.

The official `evaluation_candidate/` output remains the source for s20/s40/s60/s90 MAE/RMSE. **Do not treat per-epoch sampled validation diagnosis as four-severity test results.**

See `RELEASE_2.0.2.md` for metric definitions, caveats, and offline report reconstruction.
