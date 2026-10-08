# Mask_RAFT-STLA

Full mask-aware RAFT + STLAformer for TraffiDent incomplete sensing. This project is independent of `Baseline`; it uses only the durable `Base_SupportBench` data contract and the official STLAformer source/checkpoint managed by `Backbone`.

## Architecture

- **Mask-aware STLAformer**: zero-initialized mask context (`observed`, node coverage, time coverage, missing age) is injected into the official STLAformer embedding while preserving the pretrained model at initialization.
- **Mask-aware periodic RAFT retrieval**: periods `[4,2,1]`; Pearson similarity is computed only on observed query evidence. Scores are coverage-regularized and overlapping train windows are excluded by true `window_start` intervals.
- **Full-train memory**: queries are incomplete, retrieval keys and futures are complete Train histories/futures.
- **Reliability-aware fusion**: Top-K similarity, margin, entropy confidence and observation coverage drive period routing and a node × horizon gate.
- **End-to-end optimization**: pretrained STLAformer initialization, then warm-up → partial joint training → full joint fine-tuning. Final checkpoints contain the whole model.

## Required layout

Place this directory under `Speed_LLM_RAG/` beside `Base_SupportBench/` and `Backbone/`. The official STLAformer source must exist at `Backbone/vendor/STLAformer/model/STLformer.py`, and a formal STLAformer checkpoint must exist under `Backbone_exps/<target>/STLAformer/TraffiDent_Net_06/formal/` or be supplied with `MASK_RAFT_STLA_BACKBONE_CHECKPOINT`.

## Commands (PowerShell)

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:TRAFFIDENT_TARGET = "speed"

python -u Mask_RAFT-STLA\scripts\preflight.py --target speed
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split train
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split val
python -u Mask_RAFT-STLA\scripts\train.py --target speed --seed 10
```

Formal training now **automatically evaluates the selected `best.pt`** on `s20/s40/s60/s90` after training succeeds. Results are written under `<run_dir>/evaluation/`, including `severity_summary.csv`. The root `summary.json` is updated with `test_evaluated: true`.

Manual re-evaluation remains available:

```powershell
python -u Mask_RAFT-STLA\scripts\evaluate.py --target speed --checkpoint <best.pt>
```

To intentionally train without the automatic evaluation step, add `--skip-evaluate`.

Smoke:

```powershell
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split train --smoke --force
python -u Mask_RAFT-STLA\scripts\build_retrieval_cache.py --target speed --split val --smoke --force
python -u Mask_RAFT-STLA\scripts\train.py --target speed --seed 10 --smoke
```

For flow, replace `speed` with `flow`.


## v1.1 Gadi throughput profile
- Formal batch size: **64** (the previous hidden `min(...,16)` cap has been removed).
- DataLoader: 4 workers, pinned memory, persistent workers, prefetch=2. Smoke keeps workers=0 for portability.
- Full training future memory is preloaded to the V100 when possible; OOM automatically falls back to CPU mmap.
- PBS keeps **ncpus=12 per V100** as required on Gadi gpuvolta.
- Each formal job writes a sibling `*.gpu.csv` sampled every 30 seconds. `history.csv` and `EPOCH` logs include samples/s and peak CUDA memory.
- Optional runtime overrides: `--batch-size N --num-workers N`.
- `runtime.amp` remains false, so v1.1 changes throughput rather than numerical precision.


### Timestamped formal logs (v1.1.1)
Every line in the Gadi formal `.live.log` is prefixed with `[YYYY-MM-DD HH:MM:SS]`. The timestamping is applied at the PBS output pipeline so third-party warnings and multi-line command output are covered as well.


### Automatic formal evaluation (v1.1.2)
- Successful non-smoke training immediately evaluates the selected `best.pt` on severities `20/40/60/90`.
- Smoke runs skip formal evaluation automatically.
- `--skip-evaluate` is available for diagnostic/recovery runs.
- An evaluation failure propagates as a non-zero training/PBS exit instead of being silently ignored.
- The timestamped PBS pipeline also timestamps every automatic evaluation line.
