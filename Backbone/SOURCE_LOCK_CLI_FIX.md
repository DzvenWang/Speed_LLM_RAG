# Source-lock CLI compatibility fix

The current `scripts/source_lock.py` interface is:

```bash
python Backbone/scripts/source_lock.py verify --config Backbone/configs/backbone.yaml
python Backbone/scripts/source_lock.py create --config Backbone/configs/backbone.yaml
```

Older callers still used the retired `--root`, `--sources`, and `verify-one` arguments.  Those callers have now been updated.

This change is engineering-only.  It does not change PatchSTG weights, source commit, data contracts, evaluation caches, metrics, or prediction semantics.
