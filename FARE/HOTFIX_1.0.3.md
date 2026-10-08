# FARE v1.0.3 — Automatic post-train and manual PBS Evaluate

- Existing training PBS now enables auto-evaluation of `candidate_best.pt`
  for s20/s40/s60/s90 by default (`AUTO_EVALUATE=1`).
- Recommended `scripts/submit_gadi.py train` submits independent training and
  `afterok` dependent candidate-evaluation PBS jobs; prevents walltime conflicts.
- Both speed and flow can be scheduled with one command and separate train/eval
  checkpoint and dataset namespaces.
- Optional `--finalize` additionally schedules OOF controller fitting,
  Validation calibration, and final evaluation of calibrated `best.pt`.
- `scripts/submit_gadi.py evaluate` independently submits a manual PBS job to
  evaluate a candidate, calibrated best, or phase checkpoint; supports
  validation/test, severities, ablation mode and custom output path.
- Automatic candidate tests write `evaluation_candidate/`, calibrated best
  tests write `evaluation/`, manual runs default to `manual_evaluation/`.
- Timestamped logs, Gadi job dependency IDs and nonzero failure propagation.
- Tested with a fake `qsub` and PBS bash syntax, not with the live Gadi scheduler.
