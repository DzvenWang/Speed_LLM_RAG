# Online_PSTG v0.1.0

Strictly causal online-learning implementation for the existing PatchSTG backbone in `Speed_LLM_RAG`.

## Expected directory layout

```text
Speed_LLM_RAG/
├─ Online_PSTG/
├─ Backbone/
├─ Backbone_exps/
├─ ExperimentCache/
├─ ExperimentCommon/
├─ TraffiDent_SupportBench/
└─ data/processed/traffident/supportbench/
```

Outputs are written beside the project:

```text
Online_PSTG_exps/
Online_PSTG_logs/   # automatic console logs when using scripts/run_pipeline.py
```

## 1. Preflight

Recommended Python entry (also writes logs automatically):

```powershell
python scripts/run_pipeline.py --stage development --variant head --preflight-only
```

Direct preflight is also available:

```powershell
python scripts/preflight.py --stage development
```

## 2. Small local run

```powershell
python scripts/run_pipeline.py --stage development --variant head --max-queries 256
```

This automatically saves console logs under `Online_PSTG_logs/TraffiDent_Net_06/development/<run_tag>/`.

## 3. Development run

```powershell
python scripts/run_pipeline.py --stage development --variant head
```

## 4. Compare update capacity

```powershell
python scripts/run_sweep.py --stage development --variants frozen,head,last_block_head,full
```

## Useful overrides

```powershell
python scripts/run_online.py --stage development --variant head --replay-size 128 --min-mature-cases 8 --steps-per-update 1 --update-every 1 --learning-rate 1e-5
```

Use `--project-root` if `Online_PSTG` is not directly inside `Speed_LLM_RAG`.

## Strict causality

At current forecast origin `t`, the updater only receives past queries whose entire 12-step future has already occurred. No current/future labels are used before prediction.

With the current H=P=12 contract, maturity is computed from raw `window_start`, not from row number, so the protocol remains correct even if stream spacing changes.

## Recommended first experiment

Start with `head`, because it provides a cheap, stable online baseline. Then compare `last_block_head`. Use `full` mainly to test whether more adaptation capacity is worth the forgetting/compute cost.

## Main outputs

Each run saves:

- `summary.json`
- `stream_metrics.csv`
- `updates.csv`
- `resolved_config.json`
- `final_online.pt` (adapted variants)

`summary.json` reports static vs online metrics and improvement percentage, including chronological quarter-by-quarter gains.
