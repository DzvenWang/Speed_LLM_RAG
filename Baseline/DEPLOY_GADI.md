# Deploy Baseline v0.4.0 on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
cp -a Baseline Baseline.before_v040_$(date +%Y%m%d_%H%M%S)
unzip -o Baseline_v0.4.0.zip
cd Baseline
conda activate TFP
```

## 1. Freeze official source trees

```bash
bash scripts/setup_official_models.sh
bash scripts/strip_vendor_git.sh
python scripts/verify_official_sources.py
```

## 2. Reuse the existing v0.3.1 Development cache

```bash
python scripts/migrate_cache_v040.py --stage development
python scripts/validate_cache.py --stage development
python scripts/retrieval_leakage_audit.py --stage development
```

If no old cache exists, build a fresh Development cache instead with `qsub -v STAGE=development pbs/prepare_data.pbs`.

## 3. Project checks

```bash
python scripts/preflight.py
python scripts/smoke_new_main_runners.py
python tests/smoke_identity_contract.py
python tests/smoke_prediction_contract.py
python tests/smoke_protocol_config.py
python tests/smoke_shared_protocol_consumer.py
python tests/smoke_story_contract.py
python tests/smoke_runner_safety.py
python scripts/prepare_bridges.py development
qsub pbs/compatibility_audit.pbs
```

## 4. GPU smoke before full Development

```bash
qsub -v METHOD=GinAR pbs/methods/new_main_gpu_smoke.pbs
qsub -v METHOD=RAFT  pbs/methods/new_main_gpu_smoke.pbs
qsub -v METHOD=SARAF pbs/methods/new_main_gpu_smoke.pbs
```

Only after all pass:

```bash
bash scripts/submit_prebackbone_main_development.sh
```

## 5. Do not run yet

Do not create final ImputeFormer+forecast or matched RAST main-table results until the final Backbone is frozen. Do not materialize Test until evaluation is authorized.
