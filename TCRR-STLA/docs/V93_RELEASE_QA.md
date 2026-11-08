# TCRR-STLA v9.3.0 release QA

## Evidence of local testing

Assembled from the supplied v9.2 full source and the supplied v7.1 road graph source, preserving v9/v9.2 modules intact.

1. `pytest -q tests/v9 tests/v92 tests/test_v52_pipeline_and_cache.py tests/test_v93_*.py`: **52 passed**, with one non-fatal PyTorch legacy warning.
2. `pytest -q tests/test_v93_*.py`: **10 passed**, including full synthetic preparation → multi-view Teacher → Memmap → four-stage CPU training → checkpoint → calibration/check → auto diagnosis.
3. `python -m compileall -q scripts/train_v93.py scripts/evaluate_v93.py scripts/preflight_v93.py tcrr_stla/v93`: passed.
4. `bash -n pbs/formal_v93.pbs pbs/evaluate_v93.pbs`: passed.
5. `none`, `fixed`, `v7_edge_attention`: forward & gradient tested.
6. Freeze Normal prediction in CF stage: checked with exact probe and parameter fingerprint.
7. `supportbench.sensing` is the actual uploaded Base_SupportBench implementation in the synthetic test, with synthetic data/layouts.

## What was NOT verified here

* No genuine `TraffiDent_Net_06` data directories were available in the runtime, nor official STLAformer vendor checkpoint, nor Gadi V100.
* Formal `prepare`, `train`, `diagnose` and severity `test` on real data are **not run**; no experimental gain is claimed.
* GPURAM/throughput under full Q×V, actual speed/flow H12 and direct-mask statistics require Gadi.
* The old v5-only tests with hardcoded legacy release identities are historical; the v5.2 version assertion is updated to retain its v5-specific checks while permitting root v9.3 VERSION.

## Scientific limitations

* Direct CF Student at inference sees no extra observations; CF gain must be shown beyond independently trained residual-only and parameter-capacity-matched single-head controls.
* Historical candidate-level cross-mask utility remains implemented in preserved v9.2; v9.3 Direct student learns **query-level** cross-mask utility differences. They are different mechanisms and should not be presented as identical.
* Counterfactual teacher is a controlled STLA prediction response to masking, not evidence of physical causal change in traffic.
* Graph comparison, one-head capacity comparison, single-mask vs multi-mask and retrieval necessity require **independent retraining**. Same-checkpoint toggling does not establish component causal contribution.
