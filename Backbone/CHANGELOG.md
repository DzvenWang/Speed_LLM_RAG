# Changelog

## v1.3.0
- Added `complete`/`incomplete` historical sensing input modes to Formal/STLAformer and all Backbone models; the original incomplete mode remains the default and uses the same `formal` paths.
- Reuses frozen Base_SupportBench `x_full_norm` and `x_obs_norm` arrays; full mode masks genuinely raw-invalid history entries without leaking future observations.
- Separate complete-mode checkpoints (`formal_complete`) and evaluation artifacts (`checkpoint_evaluation/complete`) with mode metadata, per-mode pointers, preflight and smoke support, and Gadi PBS `TRAIN_DATA_MODE`.
- Added synthetic CPU tests and detailed commands in `DUAL_TRAINING_MODES.md`; no Baseline or Base_SupportBench modification required.

## v1.2.1
- `LATEST_RUN.txt` and `LATEST_CHECKPOINT.txt` are now project-root-relative POSIX paths, portable between Windows/local and Gadi.
- Pointer readers remain backward-compatible with old absolute Windows/Gadi paths and rebase `Backbone_exps/...` onto the current `SPEED_LLM_RAG_ROOT`.
- Added `scripts/refresh_latest_pointers.py` to migrate existing Formal runs without retraining.


## 1.2.0
- Removed the Backbone freeze/registration lifecycle and `scripts/freeze_checkpoint.py`.
- Formal training still writes `best.pt` and evaluates s20/s40/s60/s90 automatically.
- After successful evaluation, each model publishes `LATEST_RUN.txt` and `LATEST_CHECKPOINT.txt` under its own Formal directory.
- Downstream projects select the Backbone model explicitly; the pointers are convenience discovery only, not global model selection.
- No model architecture, training recipe, dataset split, Protocol, cache, or metric definition is changed.
