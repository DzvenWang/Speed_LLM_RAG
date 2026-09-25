# TraffiDent-SupportBench 1.2.4 schema

Canonical data remain model-agnostic. The benchmark stores complete traffic facts, Train-only normalization/dependency statistics, deterministic sensing views, Train-only historical memory, and exact protocol IDs. Learned embeddings, backbone predictions/residuals, support utility, Qwen labels and explanations remain derived artifacts.

## Canonical layers

```text
TraffiDent_Net_XX/
  core/
    traffic_2024.npy
    raw_valid_mask.npy
    nodes.csv
    normalization.json
  graph/
    adjacency.npy
    directed_neighbors.npz
    predictive_dependency.npz
    layouts/
      regional_layouts.json
      directional_layouts.json
      predictive_regional_layouts.json
      predictive_directional_layouts.json
  windows/
    train_full.npy / val_full.npy / test_full.npy
    train_development.npy / val_development.npy / test_development.npy
  views/view_manifest.csv
  memory/
    episode_window_starts_full.npy
    episode_window_starts_development.npy
    episodes.csv
    local_patches.npz
  protocols/
    track_general.json
    track_main_support.json
    track_predictive_structured.json
    track_ood_support.json
    track_support_challenge.json
    gap1_support_retrieval.json
    gap2_support_reasoning.json
    gap3_grounded_explanation.json
    resolved/*.csv
```

## Predictive dependency

`graph/predictive_dependency.npz` is fitted only on the Train interval and contains per-target predictive-neighbour IDs, normalized weights, best raw-step lag, lag minutes, lagged correlation, relation type, source centrality and target strength. It is used to define predictive masks, support-dependent targets and reasoning evidence. Correlation anchors may be thinned by `sample_stride`, but lag is always measured on the raw 5-min timeline. Validation/Test future is never used.

## Support-oriented sensing

General families are random/sensor/temporal/regional/directional. Predictive regional/directional families directly optimise the Train-only dependency-context-loss objective under topology/direction constraints. They are never selected using Validation/Test future or proposed-model forecast gain.

## Memory

Formal memory uses the full Train archive subsampled at `episode_stride=6` raw steps (30 min). Development memory is a subset tied to development Train windows. Historical future values are legal because they are already realized in Train.

## Derived challenge data

The formal Chapter-3 query-level predictive-context loss is `network_mean_context_loss = mean(per_target_context_loss)`; `aggregate_context_loss` is retained only as the legacy v1.2.2 target-quantile diagnostic.

Support Opportunity v2 uses `build_support_opportunity_features.py` for future-free Stage-A features and `fit_support_opportunity_calibrator.py` for Validation-outcome calibration. Test applies the frozen Validation calibrator and is never refitted or re-quantiled.


## Derived v1.2.2 artifacts

```text
candidate_pools/
  formal multi-view Top-K pools
  training mining pools (default K=256; hard + diverse negatives)

utility/*.npz
  gain_all
  gain_masked
  gain_support_dependent
  gain_h3 / gain_h6 / gain_h12
  safe_support

reasoning/*.csv|jsonl
  support-dependent utility pairs and structured records

explanation/*_inputs.jsonl
  generator-visible evidence only
explanation/*_labels.jsonl
  evaluator-only realised outcome labels

headroom/*.json
  Beneficial Recall@K / harmful Top-1 / oracle selection headroom
```

`forecast_gain` remains an alias of `gain_all` for v1.2.1 command compatibility.
