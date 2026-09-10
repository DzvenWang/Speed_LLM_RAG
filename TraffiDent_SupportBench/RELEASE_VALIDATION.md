# v1.2.2 release validation

The release was validated on the bundled synthetic TraffiDent-like graph before packaging.

## Canonical pipeline

- dependency lag smoke: PASS (`lag=2 -> 10 min` with 60-min correlation-anchor sampling)
- Stage 01-07 synthetic build: PASS
- sensing/audit/API smoke: PASS
- predictive Regional / general Regional context-loss ratio: **1.100x**
- predictive Directional / general Directional context-loss ratio: **1.104x**

The ratios above are synthetic validation only. Real TraffiDent ratios must be inspected from the Stage-06 Validation audit after rebuilding on Gadi.

## Derived pipeline

- multi-view candidate pool: PASS
- large training mining pool + diverse-negative quota: PASS
- Validation-only support-opportunity calibration: PASS
- residual alpha selection: PASS
- multi-granularity support utility: PASS
- support-dependent reasoning pairs/records: PASS
- explanation input/label physical separation: PASS
- contrastive input/label physical separation: PASS
- RAG headroom diagnostics: PASS

## Static checks

- Python compileall: PASS
- PBS/shell syntax: PASS
