# Support Opportunity v2 patch

## Why this patch exists

Job 178261014 showed that the v1 handcrafted score was engineering-correct but
scientifically weak as a final Support Opportunity definition:

- retrieval ambiguity was almost constant and therefore acted like an offset;
- SO correlated very strongly with normalised predictive-context loss;
- clipped Q10--Q90 family scaling saturated directional families;
- regional context loss was constant, making High-SO mathematically impossible
  for that family under the v1 score.

The canonical SupportBench and existing Val/Test candidate pools do **not** need
rebuilding.

## New design

### Stage A: future-free structural features

`build_support_opportunity_features.py`

Outputs:

- `predictive_context_percentile`: tie-aware family-wise Validation empirical
  percentile;
- `support_scope`: support-dependent target fraction;
- `memory_coverage_percentile`: Validation empirical percentile of top-1 memory
  retrieval score;
- `context_scope_interaction`;
- `retrieval_difficulty`: entropy diagnostic only;
- top-1 margin and score standard deviation as extra difficulty diagnostics.

No final SO score is hand-crafted here.

### Stage B: Validation-outcome calibration

After a numerical reference backbone is frozen, `build_support_utility.py`
stores candidate utility plus query-level headroom:

- `oracle_safe_global_relative_gain` (formal SO-v2 target);
- `oracle_safe_support_relative_gain`;
- `has_safe_support`;
- similarity-top1 support/global gains.

`fit_support_opportunity_calibrator.py` then:

1. uses only future-free features as inputs;
2. uses Validation realised safe headroom as the target;
3. creates out-of-fold Validation predictions;
4. fits one final Ridge calibrator on all Validation data;
5. freezes model parameters and the Validation score ECDF;
6. applies them to Test without Test fitting/re-quantiling.

The default SO-v2 inputs are:

- predictive_context_percentile
- support_scope
- memory_coverage_percentile
- context_scope_interaction

Retrieval ambiguity is excluded by default because it describes selection
*difficulty*, not whether useful support exists.

## Files

Replace:

- `configs/supportbench.yaml`
- `derived/build_support_utility.py`

Add:

- `derived/build_support_opportunity_features.py`
- `derived/analyze_support_opportunity_headroom.py`
- `derived/fit_support_opportunity_calibrator.py`
- `pbs/build_support_opportunity_features.pbs`
- `pbs/build_support_utility_headroom.pbs`

The old `derived/build_support_opportunity.py` may be retained as a v1 diagnostic
for reproducibility, but should not be used as the formal SO definition.

## Run order

### 1. Build v2 future-free features now

Existing candidate pools are reused by default:

```bash
qsub pbs/build_support_opportunity_features.pbs
```

To rebuild candidate pools explicitly:

```bash
qsub -v REBUILD_POOL=1 pbs/build_support_opportunity_features.pbs
```

### 2. Freeze the numerical reference backbone

Create NPZ files with the existing SupportBench prediction schema:

- Validation queries: `view_id`, `pred`
- Test queries: `view_id`, `pred`
- Train memory: `episode_id`, `pred`

### 3. Build utility, diagnose headroom, fit Val SO-v2 and freeze to Test

```bash
qsub -v QUERY_PRED_VAL=/path/val_query_predictions.npz,\
QUERY_PRED_TEST=/path/test_query_predictions.npz,\
MEMORY_PRED=/path/train_memory_predictions.npz \
pbs/build_support_utility_headroom.pbs
```

## What to inspect

The formal validation is not "does every family occupy 20% of High-SO".
Instead inspect:

1. Validation OOF relationship between SO-v2 and realised safe headroom;
2. Test `High > Medium > Low` for realised safe global relative gain;
3. Test is never re-quantiled or rebalanced;
4. High-SO family composition is reported as a diagnostic, not optimised;
5. similarity-top1 harmful/safe rate and oracle headroom remain large enough to
   justify reasoner/control experiments.

## v1.2.3 metric-semantics correction

The formal Chapter-3 predictive-context loss is now implemented everywhere as:

```text
network_mean_context_loss = mean(per_target_context_loss)
```

In v1.2.2, `SupportBench.predictive_context_metrics()` named the configured target quantile (q=0.75 by default) `aggregate_context_loss`, and Stage-A SO-v2 accidentally used that legacy diagnostic as `predictive_context_loss`. v1.2.3 keeps `aggregate_context_loss` for backward compatibility but formal derived artifacts use `network_mean_context_loss`.

Existing canonical traffic, windows, masks, Train memory and candidate pools do not need rebuilding. Rebuild Stage-A SO-v2 features before support-utility/headroom calibration.
