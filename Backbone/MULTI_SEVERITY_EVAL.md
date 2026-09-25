# Frozen Backbone severity evaluation: 20/40/60/90

Do **not** retrain the Backbone. `backbone_selection.yaml` remains the 40% training/selection config.

The `backbone_eval_s20/s40/s60/s90.yaml` files point the same frozen checkpoint evaluation code at four paired evaluation-only protocols and four isolated ExperimentCache namespaces.

After those protocols have been built:

```bash
bash submit_severity_evaluations.sh
```

Or submit one severity manually:

```bash
qsub -v SEVERITY=90 pbs/prepare_severity_test.pbs
qsub -v SEVERITY=90 pbs/evaluate_frozen_severity.pbs
```

Results are written to `Backbone_exps/frozen_test_s20`, `..._s40`, `..._s60`, and `..._s90`. Then run:

```bash
python scripts/summarize_severity_results.py
```
