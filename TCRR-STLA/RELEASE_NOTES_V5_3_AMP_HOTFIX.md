# TCRR-STLA v5.3.0 — AMP BCE training hotfix (2026-10-28)

## Incident

Gadi jobs `182373268` (speed) and `182373271` (flow) successfully built
S0/retrieval/normal-teacher/CF-replay/frozen-base caches but failed at the
first C12 training batch due to:

```
RuntimeError: torch.nn.functional.binary_cross_entropy and
 torch.nn.BCELoss are unsafe to autocast.
```

## Root cause and fix

Under FP16 AMP, C1 computed `positive_prob = sigmoid(raw_gain_logit)` and
`compute_losses` fed this probability to `F.binary_cross_entropy`. CUDA
autocast explicitly rejects that operation, even if the probability is cast
to FP32 inside the AMP region.

C1 now emits `cf_gain_logit` (unsquashed) alongside the unchanged
`cf_gain_probability = sigmoid(cf_gain_logit)` used by CFRM inference.
Training uses `F.binary_cross_entropy_with_logits(cf_gain_logit, label)`.
BCEWithLogits is autocast-safe and numerically stable.

No prediction formula, history/replay protocol, top-K, network parameters,
mask semantics, checkpoint schema, configuration, or output cache identity
has changed. **The canonical version stays 5.3.0 on purpose** so the large
already-completed Gadi caches can be reused. This is a code-level hotfix.

## Verification

The new `tests/test_v53_amp_bce_hotfix.py` tests raw-logit/probability
consistency and C12/Joint backward under CPU BF16 autocast. A CUDA FP16
variant is available and skipped automatically on CPU-only machines.
The entire test suite was rerun after the patch.

## Retry

Deploy the full ZIP over your existing `TCRR-STLA/` folder, **without
removing** `TCRR-STLA_cache/` or the shared S0 adapter. Submit new Speed and
Flow PBS pipelines, creating new run stamps. Teacher replay should log
`REPLAY_BANK_HIT` (assuming same retrieval and teacher fingerprints).
The frozen base cache may rebuild when the new run-local adapter path changes,
but the expensive ~50k-query CF replay need not be regenerated.

Before retrying, inspect the failed pipeline's dependent diagnostic/test jobs
with `qstat -u "$USER"` and cancel only those old dependent jobs that remain
queued, not any unrelated jobs.
