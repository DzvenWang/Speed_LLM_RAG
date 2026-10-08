# FARE v1.0.1 — Sparse-observability stability hotfix

## Reported failure

Net-06 speed smoke completed the retrieval epoch, then logged `gate: NaN` and
all-zero validation statistics in the recovery epoch, before failing at the
first injection minibatch with `RuntimeError: Nonfinite loss`.

## Root cause and fixes

1. `fare/recovery.py` had differentiable `sqrt(var)` at exact zero. Single
   eligible retrieval candidate or sparse visible anchors yield variance=0,
   producing infinite/NaN *gradients* although forward values stay finite.
   Clamp variances to `1e-6` before their square roots. Report uncertainty=10
   (instead of zero) for regions with no valid retrieval evidence.
2. `fare/graph.py` uses the same strictly-positive variance floor for the
   observable-context standard deviation.
3. `fare/losses.py` no longer replaces actively-supervised NaN/Inf loss entries
   with zeros. This had incorrectly selected zero-valued validation losses.
4. `fare/trainer.py` checks main intermediate predictions, every loss and every
   trainable gradient for finiteness *before* optimizer.step; errors now name
   the phase, batch, tensor/parameter.
5. `fare/evaluation.py` refuses nonfinite evaluation predictions and accepts
   `retrieval_off` output that does not include recovery or gate diagnostics.
6. Regression tests cover just one legal historical candidate, sparse mask,
   multiple optimizer steps, and deliberate NaN loss injection.

## Run cleanly

Do not resume from checkpoints created by failed v1.0.0 smoke run: some parameters
may already contain NaNs. Extract v1.0.1 over `FARE/` and launch a **new** smoke
run. Existing `FARE_exps` and `FARE_logs` need not be deleted; the new run
creates a fresh timestamped directory.

```powershell
python -u FARE\scripts\preflight.py --dataset TraffiDent_Net_06 --target speed
python -u FARE\scripts\train.py --dataset TraffiDent_Net_06 --target speed --seed 10 --smoke
```

## Verification scope

Synthetic tests exercise the actual code paths using a fake STLA backbone and
Base_SupportBench's metrics from the supplied source. Not yet verified here
against the user's real Windows/Gadi STLA vendor and Net-06 cache; the next
local smoke should validate this.
