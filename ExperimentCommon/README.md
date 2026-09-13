# Shared Experiment Contract v1

`ExperimentCommon` is the single source of truth for data/cache semantics shared by Backbone, Baseline, and future OwnModel/RAG code.

It owns:
- the single materialized `ExperimentCache` for development/formal/evaluation;
- safe TOD/DOW parsing and its fingerprint;
- cache identity and Protocol binding;
- canonical MAE/RMSE metrics;
- the prediction NPZ contract (`view_id`, `pred`).

It does **not** own model training or model selection.

## Gadi layout

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/
  ExperimentCommon/
  ExperimentCache/
  Backbone/
  Baseline/
  Protocol/
  Protocol_artifacts/
  TraffiDent_SupportBench/
```

## Prepare once, consume everywhere

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon
python scripts/prepare_cache.py --stage development --force
python scripts/validate_cache.py --stage development
python scripts/prepare_cache.py --stage formal --force
python scripts/validate_cache.py --stage formal
```

Backbone `screening` aliases `development`; Baseline uses the stage names directly.
## Direct-consumer policy

`ExperimentCache` is the only materialized cache location. Backbone, Baseline, and future OwnModel consumers must resolve this directory directly through `experiment_common.cache.cache_dir`; consumer-local `_cache` aliases/symlinks are intentionally unsupported.

