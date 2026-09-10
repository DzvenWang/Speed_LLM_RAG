# RMSE hotfix

Fixes Backbone `backbone/metrics.py`: the old `RMSE` entry called `_metric(...)` without `kind='rmse'`, so it silently used the default MAE branch. This patch computes real RMSE and adds masked/support/horizon RMSE fields. `scripts/refresh_screening_metrics.py` recomputes completed screening validation metrics from existing `best.pt` checkpoints; no retraining is required.
