# Legacy Baseline evaluation compatibility and source warning

This patch supports Baseline `1.3.10` Formal runs for Evaluate-only workflows
and makes vendor source tree SHA256 mismatch a visible **warning** rather than
an abort. Other identity and checkpoint checks remain strict.

## Installation

Extract the zip to your `Speed_LLM_RAG` repository root, replacing the
existing `Baseline/` files. The archive includes the full Baseline project,
not `Baseline_exps`, vendor Git repositories, or your runtime cache.

## Resume the original manifest on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export EVAL_MANIFEST="$(pwd)/EvalOnly/runs/20261013_165326_1e4d54e2/baseline.jsonl"
test -s "$EVAL_MANIFEST" || { echo 'Manifest not found'; exit 1; }

# Verify original progress markers are present. Do not delete them.
find "$(dirname "$EVAL_MANIFEST")/state/baseline" -name '*.done.json' | wc -l

qsub -N BL_Eval1GPU \
  -v KIND=baseline,MANIFEST="$EVAL_MANIFEST",SPEED_LLM_RAG_ROOT="$(pwd)" \
  EvalOnly/pbs/group_eval_only.pbs
```

The existing `EvalOnly/scripts/run_manifest.py` skips `.done.json` markers
and attempts only remaining entries. A batch may still exit nonzero if a
checkpoint is genuinely incompatible with the current runner; inspect the
new logs rather than bypassing checkpoint/identity checks.

## Changed files

- `Baseline/configs/baseline.yaml`: adds compatible version `1.3.10` and bumps
  current code version to `1.4.8`.
- `Baseline/scripts/source_lock.py`: warning-only on vendor tree SHA mismatch
  for `verify` and `verify-one`. `create` stays strict.
- `Baseline/VERSION` and `Baseline/CHANGELOG.md`.
- `Baseline/tests/test_legacy_version_and_source_warning.py`: unit tests.

## Review warnings

```bash
find Baseline_logs -name 'BaselineReEval.*.live.log' -print | tail -20
find EvalOnly_logs -name 'EvalBatch.baseline.*.live.log' -print | tail -10
```
