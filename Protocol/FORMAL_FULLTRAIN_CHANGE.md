# Formal full-training selector update (2026-09-11)

This source revision keeps Shared Protocol schema/version `1.0.0` for downstream compatibility.

The Formal Train selector now:

- keeps **all eligible unique `view_group_id` values**;
- selects exactly one deterministic representative sensing view per group;
- uses best-effort family balancing, not an infeasible exact five-family quota;
- leaves Development, Formal Val reuse, Evaluation/Test isolation, artifact schema, and consumer API unchanged.

The change fixes the Formal build failure where 56,379 unique groups could not be assigned to exact five-family quotas (maximum feasible exact assignment: 52,910).
