# Migration from ExperimentCommon v1.0.0

The tensor schema remains unchanged (`x_obs_norm`, `x_full_norm`, masks, targets, TOD/DOW, adjacency). The cache identity/governance layer changed.

## Old identity

```text
ExperimentCache/<dataset>/<stage>/
```

or severity-specific ad-hoc namespaces.

## New identity

```text
ExperimentCache/<dataset>/<protocol_name>/<severity>/<stage>/<protocol_stage_hash>/
```

This prevents stale caches from being silently reused after Protocol regeneration and prevents s20/s40/s60/s90 collisions.

## Protocol binding

- Primary training protocol: `protocol_s40` v2.0.1.
- Formal train assertion: exactly 50,000 rows, 50,000 unique groups, 10,000 per family.
- Evaluation access is enforced by Protocol's `EVALUATION_ACCESS_GRANTED.json` gate.

Downstream Backbone/Baseline/PreSuRe consumers must resolve cache locations via `experiment_common.cache.cache_dir` instead of constructing paths manually.
