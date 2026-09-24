# Online PreSuRe v3.3.0

**Predictive-Utility Retrieval for Online Traffic Forecasting under Incomplete Sensing**

v3.3 keeps the validated core unchanged:

```text
Slow causal Online PatchSTG
        +
Mature external RAG memory
        +
Missing-history completion
        +
Fixed safe forecast fusion
```

The only research change is **support selection**. v3.2 showed that widening the candidate pool exposes >10% completion opportunity, but similarity-based node-wise composition realizes only a small fraction. v3.3 therefore replaces final similarity ranking with a strictly causal **predictive-utility scorer**.

## Main idea

Stage 1 discovers a broad causal candidate pool:

```text
current incomplete query
    -> frozen-key / observed-match / complementarity / recency screen
    -> Top-64 mature cases
```

Stage 2 works at `node × 3-step temporal patch` granularity:

```text
for each missing node × temporal patch:
    visible query/candidate features
        -> online Utility MLP
        -> P(harmful), P(neutral), P(helpful)
        -> use / reject / abstain
        -> Top-K helpful supports only
```

The selected historical observations are aligned and used only to fill current artificial missing entries. Current observed values are hard-protected.

## Strictly causal utility supervision

v3.3 never uses the current query's future label to select support.

At query time `t`:

1. Build visible support features.
2. Generate a small batch of counterfactual support-completed forecasts **without reading `Y_t`**.
3. Store their predictions and feature vectors.

Only after the full 12-step future has matured:

```text
utility = MAE(slow forecast, Y_t) - MAE(candidate-support forecast, Y_t)
```

The matured utility becomes an online label for the scorer. The scorer therefore learns only from already-resolved historical queries.

## Utility classes

Default downstream utility label:

```text
Helpful : forecast improvement >= +0.05%
Neutral : -0.05% < improvement < +0.05%
Harmful : forecast improvement <= -0.05%
```

The online MLP predicts the three classes. Ranking uses:

```text
utility_score = P(helpful) - 1.5 * P(harmful)
```

A support is rejected when helpful probability is too low, harmful probability is too high, or utility score is below threshold. If no support passes, the patch abstains and leaves the original incomplete input unchanged.

## Temporal granularity

With `HISTORY=12`, v3.3 uses four 3-step patches:

```text
P0 = [0:3]
P1 = [3:6]
P2 = [6:9]
P3 = [9:12]
```

Different nodes and temporal patches can retrieve completely different historical cases.

## Variants

- `frozen` — frozen PatchSTG.
- `slow_only` — validated slow Online PatchSTG.
- `rag_similarity_global` — v3.1-style global Top-8 completion control.
- `rag_similarity_compositional` — same v3.3 `node × temporal patch` completion but ranked only by visible similarity/complementarity.
- `rag_utility_compositional` — **v3.3 main model**; same candidates/granularity/fusion, but final support selection uses causal learned predictive utility.

The most important comparison is:

```text
rag_utility_compositional - rag_similarity_compositional
```

because these two variants share the same Top-64 candidate pool, patch granularity, alignment, completion, and fixed gate.

## Recommended development runs

Preflight:

```powershell
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional --preflight-only
```

Short causal/utility smoke run:

```powershell
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional --max-queries 512
```

Controlled sweep:

```powershell
python scripts/run_sweep.py --stage development
```

Equivalent runs:

```powershell
python scripts/run_pipeline.py --stage development --variant slow_only
python scripts/run_pipeline.py --stage development --variant rag_similarity_global
python scripts/run_pipeline.py --stage development --variant rag_similarity_compositional
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional
```

Useful overrides:

```powershell
# Supports retained per node × temporal patch
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional --patch-top-k 1
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional --patch-top-k 3

# Utility supervision cost / warm-up
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional --utility-probes 8
python scripts/run_pipeline.py --stage development --variant rag_utility_compositional --utility-min-examples 512
```

## Outputs

Each run writes:

- `summary.json`
- `stream_metrics.csv`
- `slow_updates.csv`
- `retrievals.csv`
- `patch_supports.csv`
- `memory_events.csv`
- `evidence_metrics.csv`
- `utility_labels.csv` — matured downstream utility supervision only
- `utility_updates.csv` — online utility-scorer training
- `resolved_config.json`
- `final_slow_online.pt`
- `final_utility_scorer.pt` for the utility variant

Important summary fields:

```text
rag_increment_vs_slow_pct.masked_mae
retrieved_entry_oracle_by_k_vs_slow_pct.16.masked_mae
retrieved_entry_oracle_by_k_vs_slow_pct.64.masked_mae
utility.examples
utility.class_counts
utility.ready_fraction
utility.mean_selected_p_helpful
utility.mean_selected_p_harmful
utility.sampled_single_patch_oracle_vs_probe_slow_pct.masked_mae
query_level.final_helpful_fraction_vs_slow
```

`sampled_single_patch_oracle` is intentionally labelled as sampled: it only chooses the best intervention among the small causal probe set generated at query time and is not claimed as an exact PatchUtilityOracle@64.

## Causality boundary

A memory case enters retrieval only after its entire forecast horizon has matured. A utility label enters scorer training only after that same maturity condition. Current `x_full` and current `y` are diagnostic-only after the real prediction path has finished.

Qwen is still disabled. v3.3 first tests whether downstream predictive utility can beat similarity as the numerical support-selection objective. LLM transferability reasoning should only be added after this numerical contribution is established.
