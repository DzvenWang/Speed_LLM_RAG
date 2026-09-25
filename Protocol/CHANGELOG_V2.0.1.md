# Protocol v2.0.1 — Balanced Formal Training

This release keeps the v2.0.0 paired 20/40/60/90 evaluation design unchanged and fixes the primary s40 Formal training sampler.

## Change

- `protocol_s40/formal/train` is now exactly **50,000** views.
- Five primary sensing families contribute exactly **10,000** views each.
- `view_group_id` remains globally unique across the Formal training split.
- Formal validation still exactly reuses the frozen Development validation set (5,000 views; 1,000/family).
- Development remains 12,000 train / 5,000 val and unchanged in design.
- s20/s40/s60/s90 paired evaluation remains 5,000 views each, 1,000/family, with evaluation access locked until explicit grant.
- `balanced_unique_group_pick()` now resolves selected `(group, family)` pairs with a vectorized merge rather than repeated DataFrame scans, making 50k Formal construction practical locally and on GADI.

## Intended experiment

Primary Backbone/PreSuRe training remains at 40% missingness. The four frozen evaluation protocols are used after model freeze for severity robustness.
