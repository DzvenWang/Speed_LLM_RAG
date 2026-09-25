# GADI ExperimentCommon compatibility fix

The failing job showed:

```text
TypeError: validate_cache() got an unexpected keyword argument 'allow_locked_evaluation_validation'
```

This means the Backbone wrapper was newer than the ExperimentCommon API available on GADI.
Formal/development training does not require the evaluation-only bypass flag. v1.0.7 therefore calls:

```python
validate_cache(scfg, stage)
```

for training stages. Evaluation detects the installed ExperimentCommon function signature before using any optional keyword.

No Protocol/cache rebuild is required for this fix.
