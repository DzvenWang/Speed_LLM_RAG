# TCRR v5.3.0 — Post-training calibration / resume hotfix

## Incident
Speed job 182461062 finished C12 epochs 1–50 and Controller epochs 51–60,
then crashed in `tcrr_stla/v5/trainer.py::calibrate_controller` because it
iterated over undefined `vl` instead of the time-isolated `vl_cal` loader.

## Correction
`calibrate_controller` now iterates over `vl_cal`. This is necessary both
for successful execution and to keep threshold fitting strictly on the
calibration portion of the Validation partition. No forecasting architecture,
loss function, cache identity or model version is changed.

## Recover completed 60-epoch training

1. Install hotfix into the existing `TCRR-STLA/` tree.
2. Do not delete the original run directory, `last.pt`, `best_c12.pt`,
   `best_controller.pt` or `s0_shared_adapter.pt`.
3. Submit resume-only finalization plus a dependent combined POST job:

```bash
bash TCRR-STLA/pbs/submit_finalize_existing_run_v5.sh \
  /absolute/path/to/original/TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v5/formal/<stamp>_seed10_full \
  /absolute/path/to/original/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/<stamp>_seed10/best.pt
```

The helper reads the old `last.pt` metadata before submission, requiring
version=5.3.0, stage=controller, epoch=60. The trainer itself accepts
`--resume` and skips all 60 already-completed epochs. A 4-hour PBS
finalization job performs remaining validation/calibration/check, then
`afterok` launches the single 32GB DIAGNOSTICS+TEST job.

The helper never deletes old PBS dependencies. Cancel superseded held POST
jobs only after confirming the replacement job ID.

## Verification
- Added a calibration execution regression with two separate batches that
  fails on the previous undefined-variable bug.
- Added a fake-qsub test for the resume-only / afterok submission chain.
- Kept existing version=5.3.0, original model/config/cache unchanged.

Validation on an actual Gadi V100 and the full test suite remains distinct
from local regression tests; results depend on original checkpoint files.
