# FARE v0.2.0

**FARE: Failure-Adaptive Restoration Experts for Traffic Forecasting under Incomplete Sensing**

This is the full-strength expert version. Unlike v0.1.0, there are no heuristic placeholder experts and no Top-32 missing-node selection. Every expert estimates every `artificial_missing` cell.

## What changed from v0.1.0

- Removed linear-interpolation Temporal expert.
- Removed neighbour-average Graph expert.
- Removed single-window rank-4 SVD Low-Rank expert.
- Removed nearest-history-copy Retrieval expert.
- Added a trained graph-temporal Context expert with frozen-STLA forecast loss.
- Added causal 28-day TMCP-style Structural tensor completion.
- Added learned Context-embedding retrieval, target-aware Top-K candidate reranking, support fusion and confidence gating.
- Added a Backbone-Aware expert conditioned on the frozen STLAformer forecast.
- Added both Expert Oracle and Safe Oracle.
- Added resumable deterministic artifacts; formal runs reuse existing checkpoints/index unless `--force` is supplied.

## Runtime dependencies

The repository root must contain:

```text
Speed_LLM_RAG/
  Base_SupportBench/
  Backbone/
  FARE/
```

Baseline is not a runtime dependency.

## 1. Engineering smoke

Smoke uses the **same full architectures**, but only a small training subset, 2 epochs and a smaller retrieval index. Its metric values are not research results; it is only for runtime validation.

PowerShell:

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path

New-Item -ItemType Directory -Force -Path "FARE_logs\speed" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

python -u FARE\scripts\run_pipeline.py `
    --target speed `
    --smoke 2>&1 |
    Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.speed.smoke.$ts.live.log"
```

Expected high-level stages:

```text
FARE_PREFLIGHT_PASS
CONTEXT_EPOCH ...
TRAIN_CONTEXT_PASS ...
RETRIEVAL_EMBED ...
BUILD_RETRIEVAL_INDEX_PASS ...
RETRIEVAL_EPOCH ...
TRAIN_RETRIEVAL_PASS ...
BACKBONE_AWARE_EPOCH ...
TRAIN_BACKBONE_AWARE_PASS ...
FULL_STRENGTH_SUMMARY ...
FULL_STRENGTH_EXPERT_PASS ...
FARE_V02_PIPELINE_PASS
```

## 2. Formal full-strength training

This uses all formal train/val rows, early stopping, the 50k chronological retrieval memory and full TMCP settings.

```powershell
New-Item -ItemType Directory -Force -Path "FARE_logs\speed" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

python -u FARE\scripts\run_pipeline.py `
    --target speed `
    --stage train 2>&1 |
    Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.speed.train.$ts.live.log"
```

Artifacts are saved under:

```text
FARE_artifacts/speed/TraffiDent_Net_06/v0.2/formal/
  context/best.pt
  retrieval_index/
  retrieval/best.pt
  backbone_aware/best.pt
```

If a stage has completed, rerunning `--stage train` reuses it. Use `--force` only when intentionally rebuilding.

## 3. First research diagnostic

After full training, evaluate s40 and s90. The default is 100 balanced queries per severity.

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
python -u FARE\scripts\run_pipeline.py `
    --target speed `
    --stage eval 2>&1 |
    Tee-Object -FilePath "FARE_logs\speed\FARE.v0.2.speed.eval.s40_s90.$ts.live.log"
```

Or a single severity:

```powershell
python -u FARE\scripts\run_pipeline.py --target speed --stage eval --severity 90 --max-rows 100
```

## Decision gate before any Router

Do **not** build a learned MoE router merely because the code runs. Continue to routing only if full-strength diagnostics show real complementarity. The most important fields are:

- `best_single_gain_percent`
- `safe_oracle_gain_percent`
- `safe_oracle_over_best_single_pp`
- `safe_oracle_routes`
- per-family expert gains

A useful target is a best single expert around or above 2% and Safe-Oracle headroom materially above that (preferably another ~2 percentage points or more), with at least two experts winning on meaningful subsets/failure families. If Safe Oracle collapses to one expert, make that expert the main method instead of forcing MoE.

## Scientific safety / leakage contract

- External history for Structural and Retrieval comes from the chronological train interval only.
- Retrieval candidates must finish before the current query begins.
- `x_full_norm` is used as a training label or evaluation diagnostic only.
- `y_norm/y_raw` are used for training/validation objectives or evaluation only.
- Evaluation expert inference never receives hidden history truth or future truth.
- Only `artificial_missing` cells may be modified.
