# TCRR-STLA v5.3.0 — evaluation S0-batch hotfix (2026-10-29)

## Trigger (Gadi PBS)

- Speed POST 182502729 and Flow POST 182503074 exited at first
  `C12_four_switches` validation diagnosis with
  `ValueError: S0 adapter training settings mismatch`.
- Speed FINALIZE 182502728 and Flow FINALIZE 182503073 **both succeeded**;
  do not retrain or recalibrate these completed experiments.

## Root cause

`scripts/evaluate_v5.py` restored the saved formal configuration from
`best_c12.pt` (train batch size 64), then erroneously assigned
`cfg['training']['batch_size'] = a.batch_size` (diagnostic batch size 8).
`initial_model()` checked the S0's strict `s0_identity` and correctly refused
this tampered formal training configuration. Both evaluation and diagnostics
were broken; the cached adapter itself was valid.

## Fix

1. Keep `cfg['training']['batch_size']` exactly as saved in the trained
   checkpoint. `DataLoader` already takes `a.batch_size` directly. The
   evaluation batch size remains independently configurable without changing
   model identity, weights, the S0 fingerprint, or the experiment protocol.
2. Print `V5_EVAL_BATCH_CONFIG` (formal training batch, evaluation batch,
   evaluation workers) to disambiguate the two settings in future PBS logs.
3. Validate evaluation `--batch-size >= 1`.
4. Add `pbs/submit_postprocess_existing_v5.sh` for **POST-only** resubmission
   after successful FINALIZE. It checks the existing formal artifacts, both
   C12/C3 checkpoint stage/version, adapter SHA, S0 training identity and
   target/dataset **before submitting a job**. It submits only one combined
   DIAGNOSTICS + TEST job with default 12 CPUs, 1 V100 and 32GB RAM.

S0 version, Backbone SHA, data/mask identity and actual training-batch
validation remain strict: no bypass, regeneration, cache invalidation,
checkpoint modification, or model re-training.

## Deployment and resubmission

Unzip patch at `Speed_LLM_RAG` project root; run the new script for each
existing finalized run, e.g.:

```bash
bash TCRR-STLA/pbs/submit_postprocess_existing_v5.sh \
  "$PWD/TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v5/formal/20261028_155629_seed10_full"
bash TCRR-STLA/pbs/submit_postprocess_existing_v5.sh \
  "$PWD/TCRR-STLA_exps/flow/incomplete/TraffiDent_Net_06/v5/formal/20261028_155629_seed10_full"
```

Each returns a `POSTPROCESS_ID`; no TRAIN/FINALIZE rerun is needed.

## Verification

- Synthetic evaluation entry point at batch sizes 1, 8, 32, 64, 128 calls the
  real S0 adapter identity check with **formal training batch 64**.
- Genuine S0 training identity mismatch and data fingerprint mismatch are
  still rejected.
- Mocked Speed/Flow qsub tests verify one POST job and refuse bad metadata.
- Focused tests: 69 passed before adding POST-only launcher; updated focused
  tests and full-suite pytest assertions also passed locally.
- No real Gadi POSTPROCESS run has yet completed with this hotfix; unexpected
  downstream issues must still be investigated from subsequent logs.
