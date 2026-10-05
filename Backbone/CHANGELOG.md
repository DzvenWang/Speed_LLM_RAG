# Changelog

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
