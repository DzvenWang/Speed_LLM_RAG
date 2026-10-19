# FARE v5.0.2: Four parallel V100 producers, then FARE train/eval

Run from the repository root `Speed_LLM_RAG` with TFP active. No Backbone or Base_SupportBench source modifications required. The previous v5.0.1 Fold jobs MUST be stopped before launching with the same output directories; v5.0.2 uses new default `FARE_cache/oof_v502`.

## 1. Install & test

```bash
mv FARE FARE_v501_backup
unzip FARE_v5.0.2_Full_FourGPU_OOF_AutoFARE.zip
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/FARE:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
python -m pytest -q FARE/tests
```

## 2. CPU-only plan inspection (no PBS/GPU)

```bash
python -u FARE/scripts/submit_oof.py \
  --target both --nfolds 2 --warmup 0.30 --purge 12 \
  --epochs 100 --batch-size 16 --parallel-folds --auto-fare --plan-only
```

Check 2 folds for each target and reasonable fit/val/score counts. Plan files are immutable once written. A different plan requires a distinct `--output-dir` and separate OOF experiment.

## 3. Submit ENTIRE workflow at once

```bash
python -u FARE/scripts/submit_oof.py \
  --target both --nfolds 2 --warmup 0.30 --purge 12 \
  --epochs 100 --batch-size 16 --parallel-folds --auto-fare
```

PBS dependency DAG:

```text
speed fold 0  (1 V100, 12 CPU) ┐
speed fold 1  (1 V100, 12 CPU) ├── afterok(all 4) ── finalize OOF ── afterok ┬─ FARE formal speed (train -> eval)
flow fold 0   (1 V100, 12 CPU) ┤                   (1 V100)              └─ FARE formal flow  (train -> eval)
flow fold 1   (1 V100, 12 CPU) ┘
```

Four producers MAY run in parallel, subject to PBS scheduling; they need not be on the same physical node. Finalization exports paired predictions with real trained fold checkpoints, verifies provenance, builds memory for both targets and runs final preflight before exit 0. FARE jobs stay dependency-held until finalization passes. The FARE jobs use independent V100s, one per target, and auto-evaluate s20/s40/s60/s90 according to `formal.yaml`.

Find exact PBS IDs in the command output and the receipt:

```
FARE_exps/oof_submissions/<timestamp>_seed10.json
```

All four producers require `--fold-id` and run `FARE/pbs/oof_fold.pbs`; finalize uses `FARE/pbs/oof_finalize.pbs`, then the target FARE jobs use `FARE/pbs/full_pipeline.pbs`.

## 4. Monitor

```bash
qstat -u "$USER"
find FARE_logs/oof -type f -name '*.live.log' -print
ls -lh FARE_cache/oof_v502/{speed,flow}/TraffiDent_Net_06/fold_*/COMPLETED.json
```

Expected outputs include:

```
FARE_cache/oof_v502/speed/TraffiDent_Net_06/fold_00/{last.pt,best.pt,COMPLETED.json}
FARE_cache/oof_v502/speed/TraffiDent_Net_06/fold_01/{last.pt,best.pt,COMPLETED.json}
FARE_cache/oof_v502/flow/TraffiDent_Net_06/fold_00/{last.pt,best.pt,COMPLETED.json}
FARE_cache/oof_v502/flow/TraffiDent_Net_06/fold_01/{last.pt,best.pt,COMPLETED.json}
FARE_cache/oof_speed_paired.npz
FARE_cache/oof_flow_paired.npz
FARE_cache/speed/TraffiDent_Net_06/memory_v502_10_50000/
FARE_cache/flow/TraffiDent_Net_06/memory_v502_10_50000/
```

## 5. Alternative: manually submit FARE after OOF

If you omit `--auto-fare`, the four producers and dependent finalize job are submitted but not FARE. After `FARE502_OOF_FINALIZE_AND_PREFLIGHT_PASS`, run:

```bash
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml --target speed \
  --oos-forecast-file FARE_cache/oof_speed_paired.npz
python -u FARE/scripts/submit_gadi.py --config FARE/configs/formal.yaml --target flow \
  --oos-forecast-file FARE_cache/oof_flow_paired.npz
```

Do NOT also run these commands when `--auto-fare` has already scheduled FARE jobs, or you will duplicate training.

## 6. Resuming after 48h PBS limit

Each Fold saves `last.pt` atomically per epoch. The existence of `best.pt` ALONE does not count as completed; the `COMPLETED.json` marker plus checkpoint/protocol digests is required. If a producer fails or reaches 48h, dependent finalization and FARE jobs will not run. Do not resubmit the same DAG while prior Fold jobs remain active. Inspect receipt and PBS states, cancel any stale dependents, and resubmit the workflow with the same fold settings. Completed folds will exit quickly; incomplete folds recover from `last.pt`.

## 7. Research integrity

- Reference STLAformer formal incomplete checkpoints are used to verify target/shape and interpret the shared normalization. They are NOT used to synthesize fake in-sample OOF forecasts.
- Each producer trains on a chronological, time-purged prefix, and uses that fold checkpoint to predict both artificially incomplete and complete historical inputs. Provenance includes checkpoint hashes, data cache manifest, fold plan, training and early-stopping label end, per-fold prefix normalization.
- Every cache row is retained by distinct `row_id`/episode key; no `window_start`-only deduplication.
- Teacher/query evaluation must also respect historical episode availability; OOF alone is not a guarantee of end-to-end leakage-free retrieval.
- 100 epochs is an upper bound. No guarantee all 4 Fold jobs complete before 48h. Gadi throughput and queue delays can dominate turnaround.
