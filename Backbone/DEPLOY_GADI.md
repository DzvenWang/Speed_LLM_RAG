# Deploy Backbone v0.4.0 on Gadi

## 0. Finish any already-submitted v0.3.1 maintenance job first
If `BackboneMetricRefresh` job 178653087 is still queued/running, let it finish before replacing the project. This avoids changing source/config files underneath an already-submitted job.

## 1. Replace wrapper code (existing experiments/cache/vendor data are not deleted by unzip)
```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
cp -a Backbone Backbone.before_v040_$(date +%Y%m%d_%H%M%S)
unzip -o Backbone_v0.4.0_PatchSTG_DSTSGNN.zip
cd Backbone
conda activate TFP
```

## 2. Install PatchSTG's one new Python dependency
```bash
python -m pip install 'timm==1.0.12'
```

## 3. Extend official source lock
```bash
bash scripts/setup_official_models.sh
bash scripts/strip_vendor_git.sh
python scripts/verify_official_sources.py
```
The setup script preserves valid tree-locked v0.3.1 vendor sources and only clones a repository when it is missing/unverifiable. It adds PatchSTG and DST-SGNN at their frozen commits.

## 4. Confirm old Stage-1A and frozen Development cache are still usable
```bash
python scripts/validate_cache.py --stage screening
python scripts/check_stage1a_ready.py
python scripts/smoke_metrics.py
python scripts/smoke_recent_model_contracts.py
```
Do not rebuild Protocol/development or the screening cache.

## 5. Run one-batch GPU adapter smokes
```bash
qsub -v MODEL=PatchSTG pbs/recent_adapter_smoke.pbs
qsub -v MODEL=DST-SGNN pbs/recent_adapter_smoke.pbs
```
Both jobs must end with `RECENT GPU SMOKE PASS` and `SCRIPT_EXIT_STATUS=0`.

## 6. Submit only PatchSTG + DST-SGNN
```bash
bash scripts/submit_recent_screening.sh
qstat -u $USER
```
Logs are under `/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone_logs`.

## 7. After both finish
```bash
python scripts/summarize_screening.py
```
This applies the unchanged selection rule across all six models. Only after this summary should Formal Protocol/cache be built.
