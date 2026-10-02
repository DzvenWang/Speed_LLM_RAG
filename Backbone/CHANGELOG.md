# Backbone v1.1.10

- Unified direct/local and Gadi Formal lifecycle: Formal training now automatically evaluates the produced `best.pt` on s20/s40/s60/s90 from `run_model_formal.py`.
- Removed duplicate post-training evaluation from the PBS wrapper.
- Fixed child-process config/checkpoint path resolution in `post_training_evaluate.py`.
- Unified PatchSTG submission through `pbs/run_model_formal.pbs`.
- No model/protocol/cache/metric/hyperparameter changes; checkpoint freezing remains explicit-only.

# Backbone v1.1.8

- Gadi PBS logging fix: `timestamp_stream.py` is launched with `$TFP_ENV_ROOT/bin/python3` instead of `$TFP_ENV_ROOT/bin/python`.
- Formal model training, preflight, and evaluation continue to use the validated TFP `$TFP_ENV_ROOT/bin/python`; no model, recipe, Protocol, cache, or metric changes.
- Prevents Gadi deprecated `python` launcher failure from breaking the process-substitution log pipe with SIGPIPE/exit 141.

## 1.1.8
- Direct/local `scripts/run_model_formal.py` runs now automatically tee stdout/stderr to a per-model `.local.<timestamp>.live.log` under `Backbone_logs/<target>/<model>/` while preserving live terminal output.
- Gadi/PBS behavior is unchanged: when `BACKBONE_PBS_LIVE_LOG=1`, the Python runner does not create a second log, so the PBS job remains the single complete live log including post-training evaluation.
- No training, cache, metric, checkpoint, or evaluation semantics changed.

## 1.1.6
- Unified PatchSTG Formal submission with the other five Backbone models through `scripts/run_model_formal.py` and `pbs/run_model_formal.pbs`.
- PatchSTG keeps its dedicated, previously validated training/smoke implementation; the unified Formal entry point dispatches to it internally.
- `scripts/smoke_all_formal.py` now validates all six models through the same `run_model_formal.py` user-facing Formal entry point.
- `pbs/run_model_formal.pbs` now accepts `MODEL=PatchSTG`, so all 12 model-target Formal jobs use one PBS script.
- Formal protocol remains unchanged: seed 10, s40 training/validation, automatic s20/s40/s60/s90 checkpoint evaluation, explicit-only freeze.

## 1.1.5
- `scripts/smoke_all_formal.py` now automatically writes a complete controller + child-process transcript to `Backbone_logs/smoke_all_formal/BackboneSmokeAllFormal.<timestamp>.log` while continuing to stream the same output to the terminal.
- No change to model, Formal training, checkpoint evaluation, cache, metric, or experiment-output semantics.
- Removed packaged `__pycache__` directories and `.pyc` files.

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
