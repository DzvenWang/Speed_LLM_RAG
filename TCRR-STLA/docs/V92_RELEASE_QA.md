# v9.2.0 Release Quality Assurance

## Confirmed within this container

- Complete project includes retained v9.1, legacy v5/v4 and new `tcrr_stla/v92/` modules, scripts, configs and formal PBS launcher.
- `python -m compileall -q .` succeeded.
- `bash -n pbs/formal_v92.pbs pbs/runtime.sh` succeeded.
- `python -m pytest -q tests/v92 tests/v9` succeeded (36 passed, one warning from a previous v9.1 test).
- v9.2 synthetic CPU integration executes fake STLA teacher, Normal Bank, independently masked retrieval, both TRAIN mask views, VAL preparation, Normal/Alignment/Utility/Joint/Paired training and report generation.
- Strict Wrong-mask branch is covered by synthetic CPU monkeypatch end-to-end test. Correct- and wrong-mask results use identical query IDs.
- Same-ID paired supervision with time-legal candidates has a nonzero match rate in synthetic integration.
- Exact zero-CF invariant, no-neighbour GNN zero output, Full Teacher selection influence, Normal frozen-parameter invariance and utility validity invariance tested.
- Overall historical tests were attempted, but the full suite exceeded the execution window. Do not claim it passed.

## NOT verified here

- Actual Gadi/NCI V100 runs with 50,000 queries / full two-view historical replay.
- Real STLAformer vendor preprocessing and end-to-end GPU initial_model/adapter execution with the user's actual cache.
- Actual s20/s40/s60/s90 Test forecast metrics or guaranteed CF improvement.
- Performance and statistical value of Signed Alignment, Learned Utility, GNN and Paired losses on real speed/flow.

## Integrity and failure protections

- v9.2 teacher/retrieval bound to the exact same original STLA checkpoint.
- Source, adapter, Mask generator, retrieval code, view preparation and graph identity validated before cache reuse.
- No current Query complete input, future label or full-observation CF teacher in student forward.
- Every candidate's completed future must precede the query origin.
- Validity masks applied to loss and CF Headroom; NaN/Inf evidence rejected.
- Two views rank candidates using only their own available Query state, with common time-only anchors independent of hidden observations.
- Normal parameters frozen and compared bitwise after each CF training Epoch.
- Gradient reset every batch and logged gradient norms by module.
- No fabricated multi-GPU execution or false claims about physical graph use without supplied grounded topology.
- TEST evaluation outputs are write-once and are not involved in Checkpoint or Mechanism selection.

## Deployment caution

Use independent `v9_2` experiment/cache directories. Do not resume v9.1 checkpoints.
For research conclusions, compare `diagnostics/report.html` to independently retrained controlled ablations and locked Test after model selection. A checkpoint toggle is behavioral evidence only.
