# Deploy Backbone v0.5.0 on Gadi

Place these sibling directories under the same project root:

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/
  ExperimentCommon/
  Backbone/
  Baseline/
```

Keep the existing `Backbone/vendor/` pinned source trees if already installed. Then:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
conda activate TFP
python scripts/source_lock.py verify --root . --sources configs/official_sources.yaml
python scripts/preflight.py --stage screening
python scripts/prepare_data.py --stage screening --force
python scripts/validate_cache.py --stage screening
```

`--force` rebuilds the shared cache stage. Do not use it while another training job is reading the same shared stage.

Formal reference workflow:

```bash
python scripts/prepare_data.py --stage formal
python scripts/validate_cache.py --stage formal
bash scripts/submit_reference_formal.sh
python scripts/summarize_formal.py   # only after all seeds finish
```
