# FARE v6.0.0 submit preflight low-memory hotfix

## Symptom

`submit_gadi.py` prints `FARE5_PRE_SUBMIT_CHECK_PASS` and then the second Python import's `pynvml` warning but no `FARE5_PREFLIGHT_PASS` / PBS Job ID.

## Cause

`preflight.py` previously called `EpisodeMemory.load()` to inspect an **existing** 35,000-episode OOF Memory. That path materializes all arrays and recomputes node/capability features, so an interactive login-node pre-submit check can take a long time and use gigabytes of RAM.

## Fix

The pre-submit preflight now reads only schema-6 JSON manifest and memory-mapped NumPy array **headers**. It verifies shape, dtype, OOF provenance and Backbone hash, and streams the OOF source SHA-256 check when the source is supplied. It emits `FARE6_PREFLIGHT_*` progress milestones. The compute-node training implementation (`pipeline.prepare` and `EpisodeMemory.load(expected=...)`) remains unchanged and still loads/validates the full Memory before any training.

## Immediate action

1. If the old command is still hung, use Ctrl+C (not `qdel` unless a PBS ID has appeared).
2. Check `qstat -u "$USER"` before resubmitting.
3. Replace `FARE` with this complete hotfix ZIP (after backing up the old `FARE` folder) and run the regular command.
4. Unit tests require that `Backbone/` and `Base_SupportBench/` are accessible at workspace root.

The `--skip-precheck` flag can avoid the login-node checks if the 43/47 tests and strict model forward already passed. This submits the PBS job without checking Memory before scheduling; the PBS worker script will still execute `preflight.py` and refuse training on error. Prefer the fixed preflight for normal submissions.

**Scientific model and fresh-initialization policy are unchanged.** No old FARE weights are loaded. OOF Memory is reused.
