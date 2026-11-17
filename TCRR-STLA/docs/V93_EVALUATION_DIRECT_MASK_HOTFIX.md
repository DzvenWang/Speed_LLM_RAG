# TCRR-STLA v9.3.0 TEST direct-mask hotfix

## Reproduced Gadi error

- Speed TEST `183728958` and Flow TEST `183728969` exited with status 1.
- Both failed in `scripts/evaluate_v93.py` at old line 78, `direct &= fv`.
- PyTorch rejects in-place writes to the broadcast view returned by `direct[:,None,:].expand_as(fv)`.

## Fix

- Add `valid_direct_mask(direct, future_valid)`.
- Preserve `[B,N]` or `[B,H,N]` direct-mask input contracts.
- Intersect with `[B,H,N]` valid positions using non-in-place `direct & valid`.
- Check dimensional shape and avoid mutation of caller inputs.
- Add `tests/test_v93_evaluation_direct_mask_hotfix.py` regression tests.

## Scope

- Does not change model architecture, model weights, Teacher cache, training, or evaluation metric definitions.
- Full-source ZIP builds upon previous AMP Gradient Hotfix1; patch ZIP contains only this evaluation fix and test.
- Previous failed TEST jobs did not print `V93_TEST` results; verify `test_s20.json` etc. are absent before re-submission.
- After applying patch, re-submit existing `pbs/evaluate_v93.pbs` jobs with the same checkpoints and run directories.

## Verification

CPU regression tests for expanded two-dimensional and native three-dimensional direct masks passed; selected existing v9.3 unit tests also passed. Full Gadi TEST execution has not been performed in the patch-authoring environment.
