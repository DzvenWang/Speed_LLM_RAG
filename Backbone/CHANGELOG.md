## 1.1.4
- Keep `Backbone_logs` log-only: checkpoint-evaluation result JSONs and summary CSVs now live under `Backbone_exps/<target>/<model>/<subgraph>/checkpoint_evaluation/...`.
- Per-severity evaluation artifacts remain under `Backbone_exps`; formal smoke follows the same layout.

# Changelog

## 1.1.3
- Extended `smoke_all_formal.py` from training-only to Formal train + s20/s40/s60/s90 evaluation smoke for all six models and speed/flow.
- Added `--smoke-test-batches` to checkpoint evaluation without changing normal full evaluation behavior.
- Completed the PatchSTG smoke checkpoint contract so it can be consumed by the standard checkpoint evaluator.
- Smoke evaluation uses the same bounded metadata subset as its prediction tensors.


## v1.1.1 Slim
- Standardized runtime logs under `Backbone_logs/<target>/<model>/`.
- Added `scripts/submit_formal.sh` as the safe Gadi submission entry point.
- The submitter creates the log directory before `qsub`, so PBS `.OU` output never depends on a directory created after job start.
- PBS `.OU`, parent `.live.log`, and PatchSTG internal training log now share the same target/model directory.
- Training, automatic s20/s40/s60/s90 evaluation, model code, cache routing, and explicit-only freeze behavior are unchanged.

## v1.1.0 Slim
- Reduced the project to the active six-model Formal pipeline.
- Preserved Speed/Flow routing, seed10, s40 Formal training, automatic s20/s40/s60/s90 evaluation, and explicit-only freeze.
- Removed legacy screening/selection and automatic model-selection code.
- Removed obsolete frozen-test/evaluation paths and duplicate severity configs.
- Removed STDN_Special diagnostics, layout-recovery utilities, legacy cache helpers, historical submission wrappers, and obsolete migration notes.
- Removed generated `__pycache__` files.
- Kept model/vendor algorithms unchanged.

## v1.0.15
- Added unified 12-case local Formal smoke tests for six models x Speed/Flow.
