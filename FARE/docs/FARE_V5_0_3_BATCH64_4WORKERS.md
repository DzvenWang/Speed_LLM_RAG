# FARE v5.0.3: 4 separate V100 jobs, OOF -> FARE formal

This is the entire FARE project, not an overlay. It uses the existing Backbone and Base_SupportBench directories at the workspace root. FARE model and data schema are unchanged from v5.0.2; the OOF scheduling defaults and data loading are faster. Tested locally with synthetic data and mocked PBS submission; **real Gadi V100 throughput and batch-64 GPU memory have not been measured**.

## Prerequisites

Run from `Speed_LLM_RAG` and activate `TFP` before submitting. Verify that the existing STLAformer formal *incomplete* checkpoints and the Base_SupportBench formal caches remain present. Do not substitute the formal_complete checkpoints as historical OOF folds.

```bash
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
python -m pytest -q FARE/tests
```

## 1. Plan, CPU only (no GPU job)

```bash
python -u FARE/scripts/submit_oof.py \
  --target both --nfolds 2 --warmup 0.30 --purge 12 \
  --epochs 100 --batch-size 64 --workers 4 \
  --parallel-folds --auto-fare --plan-only
```

This freezes separate chronological plans for speed and flow under `FARE_cache/oof_v503/`. Do not change fold parameters or cache content after the plan is frozen. Use a different `--output-dir` for any genuinely different OOF experiment.

## 2. Single command submits SEVEN PBS jobs

```bash
python -u FARE/scripts/submit_oof.py \
  --target both --nfolds 2 --warmup 0.30 --purge 12 \
  --epochs 100 --batch-size 64 --workers 4 \
  --parallel-folds --auto-fare
```

The command creates:

| PBS jobs | Resources/job | Dependency |
| --- | --- | --- |
| OOF_S0, OOF_S1, OOF_F0, OOF_F1 | 1 x V100, 12 CPU, 32 GB | Independent; may run simultaneously |
| FARE503_OOF_FINAL | 1 x V100, 12 CPU, 32 GB | `afterok` all four producers |
| FARE5_speed, FARE5_flow | 1 x V100, 12 CPU, 32 GB | `afterok` successful OOF finalization |

These **are four independent single-GPU jobs**, not a single request for four GPUs in the same node. If the scheduler starts fewer than four simultaneously, that is resource scheduling, not script serialization. One GPU per formal target; formal evaluation runs automatically after training.

The submit script prints each PBS ID and writes a receipt to `FARE_exps/oof_submissions/<stamp>_seed10.json`.

## 3. Inspect progress and GPU memory

```bash
qstat -u "$USER"
find FARE_logs/oof -name '*.live.log' -type f -print
find FARE_cache/oof_v503 -name COMPLETED.json -print
```

Epoch log fields include `MAE=`, `elapsed_s=`, `train_s=`, `peak_allocated_gb=`, `batch_size=64`, and `workers=4`. CUDA allocated memory is a PyTorch measurement, **not** total device usage or all GPU allocations; retain headroom on a 32-GB V100. If batch-64 OOM occurs, cancel/stabilize the active DAG and launch a new experiment directory at batch 32. Never change batch size in-place on a partially trained fold: its optimizer-state resume is intentionally strict.

## 4. FARE formal commands if *not* using `--auto-fare`

If `--auto-fare` was omitted, wait for successful finalization and both paired files, then submit these **two independent jobs**:

```bash
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml \
  --target speed --oos-forecast-file FARE_cache/oof_speed_paired.npz
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml \
  --target flow --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

With `--auto-fare` **do not submit these jobs again**.

## 5. Completion outputs

```text
FARE_cache/oof_v503/{speed,flow}/TraffiDent_Net_06/fold_0{0,1}/last.pt
FARE_cache/oof_v503/{speed,flow}/TraffiDent_Net_06/fold_0{0,1}/best.pt
FARE_cache/oof_v503/{speed,flow}/TraffiDent_Net_06/fold_0{0,1}/COMPLETED.json
FARE_cache/oof_speed_paired.npz
FARE_cache/oof_flow_paired.npz
```

Finalization validates the provenance, temporal embargo, identity-safe episode mapping and paired predictions, builds new failure-conditioned memory, and runs CPU preflight on both targets. Formal training then runs with the unchanged `FARE/configs/formal.yaml` (FARE batch 32; separate from STLAformer fold batch 64).

If any producer exceeds the 48-hour PBS walltime, the `afterok` dependents **will not run**. Inspect the per-fold `last.pt` and submit again after cancelling/removing stale dependent jobs. Resumption restores optimizer and scheduler and never treats a partial `best.pt` as complete.

## Scientific notes

- Fold validation and training are strictly earlier than scored windows; the train-only OOF data have a purge gap. Full model training/validation/test protocols are unchanged.
- Same producer evaluates artificially incomplete and historical complete observations. 'Complete' means complete *historical input*, not an Oracle future.
- `row_id` plus Episode Key preserves multiple failure-family rows even when they have identical `window_start`.
- No promise of a fourfold walltime speedup: the two folds have different training sample counts, and Gadi queueing plus throughput depend on allocation.
