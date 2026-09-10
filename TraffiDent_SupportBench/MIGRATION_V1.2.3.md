# Migration to TraffiDent-SupportBench v1.2.3

## What changed

This release fixes a metric-semantics mismatch found by the predictive-mask audit. Chapter 3 defines the query-level predictive-context loss as

```text
C_q^loss = mean_n C_{q,n}^loss
```

but v1.2.2 `predictive_context_metrics()` exposed a q=0.75 target quantile under the ambiguous name `aggregate_context_loss`, and several derived scripts used it as if it were the network mean.

The formal metric is now:

```text
network_mean_context_loss
predictive_context_loss   # alias of the formal metric in derived artifacts
```

The old quantile remains available as:

```text
context_loss_quantile_value
legacy_aggregate_context_loss
aggregate_context_loss     # backward-compatible legacy alias
```

## What does NOT need rebuilding

Do not rebuild:

- canonical traffic/core;
- chronological splits or forecast windows;
- fixed sensing views;
- Train-only predictive dependency;
- Train historical support memory;
- existing Val/Test candidate pools.

The source-contribution normalisation correction is a uniform scalar for each network, so it does not change greedy predictive layout identities.

## What SHOULD be regenerated

Regenerate artifacts that consumed the old ambiguous metric, starting with SO-v2 Stage A:

```bash
qsub pbs/build_support_opportunity_features.pbs
```

Then rerun the predictive-mask audit:

```bash
qsub pbs/audit_predictive_structured_masks.pbs
qsub -v SPLIT=test pbs/audit_predictive_structured_masks.pbs
qsub -v SPLIT=test,DISTRIBUTION=ood pbs/audit_predictive_structured_masks.pbs
```

The OOD audit now reads canonical views directly and no longer requires an OOD Stage-A feature CSV.

After these checks, freeze the numerical reference backbone and proceed with support utility/headroom and SO-v2 Stage B.
