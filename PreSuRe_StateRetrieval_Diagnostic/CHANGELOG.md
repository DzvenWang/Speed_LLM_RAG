# Changelog

## v0.9.0 — Fragment Retrieval & Composition Diagnostic

- Refocused the diagnostic on retrieval for **incomplete-sensing completion** only.
- Added `srd/fragment_completion.py`.
- Added structured fragment partitions: whole patch, temporal blocks, node tracks, node×time blocks, and cell-level ceiling.
- Added hidden-history completion Oracle that never uses future forecasting targets.
- Added inference-available fragment-local observed-context reranking.
- Added fragment-local softmax composition across multiple historical sources.
- Added end-to-end frozen PatchSTG evaluation of every composition scheme.
- Added `scripts/15_run_fragment_completion_diagnostic.py`.
- Added `scripts/16_summarize_fragment_diagnostic.py`.
- Added `scripts/run_v09_from_existing.py` to reuse the existing Top-50 patch retrieval result.
- Added terminal + file logging through the v0.9 runner.
- Added v0.9 regression tests; full project test suite passes.

v0.7/v0.8 utility-selector code is retained only for experiment reproducibility and is no longer the primary research path.
