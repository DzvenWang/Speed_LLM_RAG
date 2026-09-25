# Simple Evaluation policy (v1.0.6)

Checkpoint evaluation is intentionally lightweight.

Hard failures are limited to:
- checkpoint/cache missing;
- checkpoint missing state_dict/model_config/mean/std;
- node/history/horizon tensor shape incompatibility;
- strict PyTorch state_dict load failure.

The following no longer block evaluation:
- Protocol evaluation grant files;
- cache/run identity/revision mismatches;
- formal-vs-evaluation identity coupling;
- historical PatchSTG spatial-layout fingerprint mismatch.

PatchSTG is constructed from the **current SupportBench nodes/adjacency** and then the
checkpoint state_dict is loaded, matching the original Backbone evaluation behaviour.
A historical/current layout mismatch is written as a WARNING in the log and summary.
