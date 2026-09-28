# Support subset audit

`audit_support_subsets.py` inspects already frozen Protocol evaluation masks and does **not** modify `FROZEN.json`, `stage_manifest.json`, view CSVs, or mask NPZ files.

Status meanings:

- `NONEMPTY`: the family has at least one support-dependent target in the frozen split.
- `N/A_STRUCTURAL`: the support threshold is mathematically unattainable for all queries in the family under the concrete sensing mask. This is expected for current `s20 temporal_outage` because max per-source missing fraction is 0.25 while the support threshold is 0.40.
- `EMPTY_ATTAINABLE`: the threshold is attainable but the aggregate family support mask is still empty. The audit fails in this case because it may indicate a data/protocol bug.
