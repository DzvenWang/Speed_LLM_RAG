# Changelog

## v0.2.4

- Replaced scalar-alpha dense-gate supervision with Forecast Intervention Landscapes.
- Added query-level utility-curve regression + best-alpha classification + pairwise alpha ordering.
- Added cell-level restoration reliability supervision and normalized reliability allocation.
- Removed mean-gate-collapse design from all three neural experts.
- Preserved Retrieval pairwise candidate forecast-utility ranking.
- Structural calibration now stores its full validation alpha utility landscape.
- Added per-query selected-alpha / reliability diagnostics in evaluation.
- New isolated artifact/experiment roots under `v0.2.4`.
- Retains A400/V100 adaptive retrieval embedding micro-batching and JSON-safe history writing.
