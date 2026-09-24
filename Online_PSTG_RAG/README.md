# Online_PSTG_RAG v0.1.0

Strictly causal online RAG for the existing frozen PatchSTG backbone.

## Expected layout

```text
Speed_LLM_RAG/
├─ Online_PSTG_RAG/
├─ Backbone/
├─ Backbone_exps/
├─ ExperimentCache/
├─ ExperimentCommon/
├─ TraffiDent_SupportBench/
└─ data/processed/traffident/supportbench/
```

Outputs are written beside the project:

```text
Online_PSTG_RAG_exps/
Online_PSTG_RAG_logs/
```

## 1. Preflight

```powershell
python scripts/run_pipeline.py --stage development --preflight-only
```

## 2. Small smoke experiment

```powershell
python scripts/run_pipeline.py --stage development --max-queries 256
```

## 3. Main development run

```powershell
python scripts/run_pipeline.py --stage development
```

Default RAG settings:

- recent memory: 512 matured cases;
- Top-K: 8;
- retrieval: 70% latent + 15% temporal context + 15% recency;
- recency half-life scale controlled by `recency_tau_steps=288` (about one day at 5-min resolution);
- residual fusion alpha: 0.35;
- correction only on directly masked nodes;
- deterministic confidence gate enabled.

## Useful overrides

```powershell
python scripts/run_pipeline.py --stage development --top-k 5 --memory-size 256 --alpha 0.25
```

Disable the confidence gate for an ablation:

```powershell
python scripts/run_pipeline.py --stage development --no-gate
```

Apply the correction to all nodes rather than only directly masked nodes:

```powershell
python scripts/run_pipeline.py --stage development --scope all
```

## Small parameter sweep

```powershell
python scripts/run_sweep.py --stage development --max-queries 1000
```

## What the memory stores

A case does **not** enter memory immediately after prediction. It enters only after its whole 12-step future has matured. At that time the project stores the frozen PatchSTG residual:

```text
residual = true future - frozen PatchSTG prediction
```

Later queries can retrieve that residual as current-regime evidence. No future label is available before its causal maturity time.

## Main outputs

Each experiment saves:

- `summary.json` — frozen vs Online-RAG metrics, gains and chronological segments;
- `stream_metrics.csv` — query-level base/RAG errors, memory size, gate and correction strength;
- `retrievals.csv` — retrieved case IDs, similarity/context/recency scores and weights;
- `memory_events.csv` — exact time each case became eligible for memory;
- `resolved_config.json`.

Logs are automatically written under:

```text
Online_PSTG_RAG_logs/TraffiDent_Net_06/development/<run_tag>/run.log
```

## Oracle diagnostic

`summary.json` also reports a Top-K oracle diagnostic. This is **evaluation only**: after the current query has already been predicted, it asks whether any individually retrieved mature residual would have improved that query. The oracle never changes the real RAG forecast.

Interpretation:

- RAG gain > 0 and Oracle clearly larger: retrieval memory contains value, but fusion/selection can improve.
- RAG ~ 0 and Oracle also ~ 0: the recent memory itself is not useful enough.
- RAG negative but Oracle positive: retrieval/fusion needs redesign before adding an LLM.

## Comparison target

The current `Online_PSTG last_block_head` development experiment achieved roughly +1.28% masked-MAE gain. Online-PSTG-RAG v0.1 should first establish whether dynamic memory can approach or exceed that gain while keeping PatchSTG frozen.
