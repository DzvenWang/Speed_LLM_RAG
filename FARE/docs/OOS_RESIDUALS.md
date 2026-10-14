# Chronologically out-of-sample (OOS) Backbone residuals

## Why it matters

A strong STLAformer fitted on a given Train window often has unrealistically small in-sample forecast residuals. A historical residual memory based on these values is not automatically an honest estimate of out-of-sample error. This impacts both scientific attribution and conditional transfer.

## Source fold manifest

`FARE/scripts/build_oos_predictions.py` reads a JSON object:

```json
{
  "folds": [
    {
      "checkpoint": "/absolute/path/to/stla_pretrained_fold1.pt",
      "trained_until": 20000,
      "evaluate_from": 20001,
      "evaluate_until": 30000
    },
    {
      "checkpoint": "/absolute/path/to/stla_pretrained_fold2.pt",
      "trained_until": 30000,
      "evaluate_from": 30001,
      "evaluate_until": 40000
    }
  ]
}
```

These values are **examples only**, not dataset-supported real folds. Each producer checkpoint must have actual training cutoff metadata verified separately. `trained_until < evaluate_from` is checked. Only real, eligible Train window starts enter the external OOS file. Earliest windows without any eligible producer can be excluded from residual memory.

The output `.npz` includes `window_start`, `prediction [M,P,N]` (normalized, same target normalization) and `trained_until` arrays. `EpisodeMemory` validates per-row cutoff and exact shape. The provenance file includes source checkpoint hashes.

If the OOF producer was trained with a different target normalization or sensing protocol, predictions are not compatible. Correct these before use; the generated file must not be presented as strictly comparable otherwise.

## Exploratory fallback

For initial development only, `pilot.yaml` chooses `checkpoint_in_sample` and emits a RuntimeWarning. `formal.yaml` explicitly rejects such memory. The production paper should use auditable chronology or clearly state its limitation rather than report fictitious OOF residuals.
