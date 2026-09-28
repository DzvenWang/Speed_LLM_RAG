# Baseline v1.1.1 PatchSTG bridge hotfix

The v1.1.0 bridge called `Backbone/scripts/source_lock.py verify-one`, but Backbone v1.0.10 supports only `verify` and `create`. This caused PatchSTG-dependent Formal runs to stop before training.

The fix changes the bridge to:

```bash
python Backbone/scripts/source_lock.py verify --config Backbone/configs/backbone.yaml
```

and reads the live Backbone model/config version from `configs/backbone.yaml`.

No Protocol or ExperimentCommon cache rebuild is required. After copying the patch, rerun:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
PY=/scratch/mh20/zw1165/envs/TFP/bin/python
$PY scripts/preflight.py --stage formal --method ImputeFormer
bash scripts/submit_primary_formal_seed10.sh
```
