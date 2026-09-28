# Support subset audit

The canonical support metric remains unchanged. In particular, the global `support_dependent_missing_threshold=0.40` is not lowered.

The new cache audit distinguishes:

- `NONEMPTY`: canonical Support_MAE is defined;
- `N/A_STRUCTURAL`: no support target can exist under the concrete sensing mask and fixed SupportBench threshold;
- `EMPTY_ATTAINABLE`: suspicious empty subset even though the threshold is attainable; the audit fails.

For current `s20 temporal_outage`, the first case is impossible by construction: each affected source is missing for 3/12=0.25 of the history, so dependency context loss (a convex weighted average of source missing fractions) cannot reach 0.40. Reporting Support_MAE as N/A is therefore the comparison-safe behavior.

The audit does not alter cache arrays, cache schema, protocol stage hashes, or checkpoint compatibility.
