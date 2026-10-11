# FARE v2.0.2 — Gadi Resource Telemetry and Online Diagnostics

Based on **FARE v2.0.1** (full-training optimizations). This is an **observability/diagnostic release**: no architecture parameters, forecast objective, memory size, Top-K, evaluation protocol, or phase-epoch budget has been changed. New log statistics do not feed training features and do not touch inference predictions.

## New monitoring during formal PBS jobs

- A separate Python resource sampler begins before training and persists through **automatic evaluation**. Sampling defaults to **every 30 seconds** (`FARE_RESOURCE_INTERVAL=30`). It uses Linux `/proc` and `nvidia-smi`, with no extra required packages.
- Records job-process CPU cores (CPU-seconds per wall-second), number of processes, combined RSS, disk IO counters, GPU utilization, VRAM utilization/usage, matching CUDA-process VRAM, power and GPU temperature.
- **Job attribution:** the sampler reports GPU readings only when `nvidia-smi` exposes a compute-process PID belonging to the tracked job's process tree. Missing instrumentation is `null` rather than a guessed 0% usage.
- Per-job files next to the normal live log:
  - `*.resources.jsonl` (flush every sample; survives walltime cancellation)
  - `*.resources.csv`
  - `*.resources.summary.json` (normal shutdown)
  - `*.resources.pbs.txt` (best-effort `qstat -f` snapshot)
  - `*.resources.monitor.log` (monitor errors, if any)
- A PBS EXIT trap terminates the sampler and creates a summary after successful or failed Python training. If PBS kills the job abruptly, the appended JSONL remains and can be summarized later.

## New training-stage diagnostics

In `FARE_exps/.../<run>/diagnostics/`:

- `epoch_diagnostics.jsonl` / `epoch_diagnostics.csv`: separated train/val aggregates for each phase and epoch.
- `epoch_resource.csv`: independent GPU/CPU samples aligned with recorded epoch UTC start/end times.
- `telemetry_report.json` / `telemetry_report.md`: combined diagnostic reports and cautious, rule-based investigation flags.

The normal `history.jsonl`, `phase_summary.json` and Stage Evaluation remain intact. New tagged stdout lines: `FARE_PROGRESS`, `FARE_DIAG`, `FARE_RESOURCE_EPOCH`, `FARE_STAGE_DIAG`, `FARE_REPORT_PASS`.

Main intermediate diagnostics (sampled every 50th batch, up to 160 batches/epoch, no extra forward):

- **Forecast attribution**: full/base/candidate MAE and gain, candidate/final helpful/harmful fraction, plus direct-masked vs other locations (train/val only).
- **Retrieval**: valid-group and candidate fractions, Top-K weight entropy, effective Top-K, dominant weight, plus existing ranking selected/oracle/uniform errors and forecast-utility teacher coverage.
- **Recovery/Control**: state and innovation uncertainty, support-token magnitude, gate mean/near-zero/near-limit fraction, harm after gate activation, utility-predicted-positive precision.
- **Performance per epoch**: training and validation samples/sec, GPU peak allocated/reserved bytes from PyTorch, process peak RSS, stage train/val/refresh duration, batch size and worker count.
- **Heartbeats**: `FARE_PROGRESS` every 500 training or validation batches, without CUDA synchronization. The reported heartbeat elapsed wall time is approximate; use epoch timings for actual throughput.

### Scientific limitations

The online MAE metrics are **current train/validation protocol metrics** (normally s40). They do **not** substitute for official full-severity (s20/s40/s60/s90) validation/test. Test outcomes cannot be used for model selection. The ranker's training labels remain training-only. Extra diagnostics access labels **only to compute monitoring statistics** and are never used in `model.forward`.

The sampled diagnostics are not exhaustive. A GPU may be busy between samples, and short bursts will be missed. Summed RSS of Python/DataLoader processes may count shared pages more than once. The performance impact of diagnostic sampling has not been measured on Gadi yet.

## Gadi usage

Defaults, `BATCH_SIZE=16`:

```bash
qsub -N FARE_v202_speed -v TARGET=speed,SEED=10 FARE/pbs/train_formal.pbs
qsub -N FARE_v202_flow  -v TARGET=flow,SEED=10  FARE/pbs/train_formal.pbs
```

For the user's `BATCH_SIZE=32` experiments:

```bash
qsub -N FARE_v202_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32 FARE/pbs/train_formal.pbs
qsub -N FARE_v202_flow  -v TARGET=flow,SEED=10,BATCH_SIZE=32  FARE/pbs/train_formal.pbs
```

Optional sampling controls:

```bash
qsub -N FARE_speed -v TARGET=speed,SEED=10,BATCH_SIZE=32,FARE_RESOURCE_INTERVAL=60,FARE_DIAG_INTERVAL=100,FARE_DIAG_MAX_BATCHES=100,FARE_PROGRESS_EVERY=250 FARE/pbs/train_formal.pbs
```

Use `FARE_RESOURCE_MONITOR=0` or `FARE_DISABLE_DIAGNOSTICS=1` to disable each independent feature if necessary.

To reconstruct a report for an interrupted job:

```bash
python -u FARE/scripts/resource_monitor.py \
  --output-prefix FARE_logs/speed/TraffiDent_Net_06/<run>.resources --summarize
python -u FARE/scripts/report_telemetry.py \
  --run-dir FARE_exps/speed/TraffiDent_Net_06/<run> \
  --resource-prefix FARE_logs/speed/TraffiDent_Net_06/<run>.resources
```

## Compatibility and verification

- Python compile + PBS `bash -n`.
- 43 tests passed locally with a **test-only** `experiment_common.metrics` stub (the external Base_SupportBench implementation is absent here). No metric stub is shipped.
- Hardware monitor self-test on Linux completed and created sampled JSONL/CSV/summary; GPU statistics were appropriately unavailable in the non-GPU container.
- Still requires Net-06 real-data smoke and Gadi V100 monitoring verification before claiming measured runtime overhead, final MAE/RMSE or improvement over v2.0.1.
- No learned parameter or checkpoint key changes. Older v2.0.1 checkpoints remain loadable by parameter shape. For `--resume`, checkpoint-embedded configuration remains authoritative; diagnostic controls default on unless explicitly disabled.
