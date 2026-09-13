# STDN_Special changelog

## 4.1.0 — 2026-09-12
- Removed ownership of the Backbone temporal parser.
- `time_features.py` is now a compatibility re-export of `backbone.time_features`.
- Removed the local Backbone parser installer and patch file.
- Cache audit failure now instructs rebuilding through `scripts/prepare_data.py --force`.

## 4.0.0 — 2026-09-12

- Replaced broad recipe/function scanning with exact in-memory `cfg.models.STDN` injection through `backbone.utils.load_yaml`.
- Fixed the previously ineffective `patience` override.
- Removed unsafe generic `batch` argument monkey patching.
- Added train-entry and actual optimizer assertions.
- Added strict TOD/DOW cache audit and the `TOD=0, DOW=3` Unix-epoch-index signature detector.
- Added a backed-up installer for the shared Backbone time parser.
- Added a synthetic regression test reproducing the numeric-index-before-timestamp bug.
- Changed the default diagnostic queue to E2 (patience attribution) + E5 (weight-decay attribution).
