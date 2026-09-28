# Support-subset semantics: mild temporal outage

`support_dependent_target_mask` is defined by the fixed Train-only predictive-context criterion:

- dependency context loss >= `dependency.support_dependent_missing_threshold` (currently 0.40), and
- target observed fraction >= `dependency.support_dependent_target_observed_min` (currently 0.50).

For the current `s20 temporal_outage` construction (`H=12`, temporal node fraction 0.75), the outage duration is 3 history steps. A predictive source is therefore missing for at most `3/12 = 0.25` of its history. Because `dependency_context_loss` is a convex weighted average of predictive-source missing fractions, its theoretical maximum is <= 0.25. It cannot reach the 0.40 support threshold.

Therefore an empty `s20 temporal_outage` support subset is **structural N/A**, not a model failure or cache bug. Do not lower the global threshold or reshape the sensing family merely to force a metric value; either would change the benchmark semantics and invalidate comparison with already frozen results.

The patch adds explicit applicability diagnostics and audit scripts. Unexpected empty subsets are still treated as errors when the threshold is structurally attainable.
