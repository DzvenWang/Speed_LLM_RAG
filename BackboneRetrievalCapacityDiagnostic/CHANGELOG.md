# Changelog

## 0.1.2
- Fixed bounded-smoke alignment when full Formal validation/test prediction artifacts contain more rows than the requested diagnostic subset.
- Prediction rows are now selected by `view_id` whenever IDs are present, so reordered full artifacts are also handled safely.
- Added an explicit prefix-only fallback for legacy prediction artifacts without IDs.
- Added regression coverage for ID-based subset selection.

## 0.1.1

- Fixed `direct_mask` shape handling. Base_SupportBench stores `direct_mask` as `[Q,N]`; forecast arrays are `[Q,P,N]`. Metrics and masked oracle now broadcast the node-level direct mask across all forecast horizons.
- Added a shared mask-broadcast helper with explicit shape validation.
- Updated synthetic smoke to reproduce the real cache contract.

## 0.1.0

- Initial Backbone–Retrieval Capacity Diagnostic implementation.
