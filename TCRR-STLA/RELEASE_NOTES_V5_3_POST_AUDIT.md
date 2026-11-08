# TCRR v5.3.0 — post-calibration follow-up audit

Scope: audit against v5.3.0 Calibration & Resume Hotfix. No model, training loss,
checkpoint format, schema, model version or cache fingerprint changes.

## Verified
- Manual static scope check of v5 model/train/evaluation/diagnostics Python modules:
  no additional undefined global symbol was found (not a substitute for real GPU execution).
- Focused calibration, resume, combined PBS and automatic diagnostics tests.
- Synthetic end-to-end Epoch-60 restoration through stage comparison,
  calibration, Locked Check and summary writing (injected small dummy data).
- POSTPROCESS already stops if diagnostics fail, before TEST.

## Follow-up correctness fix
The combined POSTPROCESS script previously required `best_c12.pt`, but silently
skipped a missing `best_controller.pt` in its stage loop. Consequently a formal
C12+C3 60-epoch run could be reported successful after testing C12 only, even
though C3 was mandatory. It now checks four nonempty formal artifacts BEFORE
running expensive postprocessing:

- `best_c12.pt`
- `best_controller.pt`
- `summary.json`
- `validation_partitions.json`

This patch makes incomplete model finalization a hard early error. It does not
alter the TRAIN/FINALIZE Python or existing checkpoints and keeps version=5.3.0.

## Deployment
Extract the incremental patch into the Speed_LLM_RAG project root, so the file
is at `TCRR-STLA/pbs/postprocess_v5.pbs`.
Do not delete current S0/replay caches, completed training checkpoints, or
already submitted PBS jobs. Note: if a PBS job has already started executing
its copied script, replacing the source script does not alter that running job.

A full Gadi-backed POSTPROCESS test is still required; synthetic and mocked-PBS
tests do not certify dataset availability, memory use, or actual V100 runtime.
