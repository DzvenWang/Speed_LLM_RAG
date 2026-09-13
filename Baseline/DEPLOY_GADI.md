# Deploy Baseline v0.6.0 on Gadi

Deploy `ExperimentCommon`, `Backbone`, and `Baseline` as sibling directories under `/scratch/mh20/zw1165/Speed_LLM_RAG/`.

Keep existing pinned `Baseline/vendor/` trees where applicable. Verify the project and shared data contract:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP
python scripts/source_lock.py verify --root . --sources configs/official_sources.yaml
python scripts/preflight.py --stage development
python scripts/prepare_data.py --stage development
python scripts/validate_cache.py --stage development
```

Formal cache can be prepared from either Backbone or Baseline because both point to the same ExperimentCache stage:

```bash
python scripts/prepare_data.py --stage formal
python scripts/validate_cache.py --stage formal
```

Do not run `--force` on a shared cache stage while active jobs are reading that stage.
