# Multi-Severity Protocol v2

## Primary research setting

- Train/development/formal: 40% missingness.
- Frozen paired Test: 20%, 40%, 60%, 90%.
- Test size: 5,000 per severity, exact family balance (1,000 × 5 families).
- Query identity and family assignment are identical across severities.

## Pairing invariant

For pair index `i`, all four severity manifests share:

- `protocol_view_id` / `pair_id`
- `view_group_id`
- `window_start`
- sensing `family`
- `distribution`

Only `view_id`, `severity` and the resulting mask differ.

## Leakage control

The Test manifest remains frozen and paired across severities. Protocol v2.0.2 does not use an access-grant file; consumers may read evaluation views directly after frozen/hash validation succeeds.
